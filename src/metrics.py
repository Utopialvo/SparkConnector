from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from typing import Optional, Dict, List
from pyspark.storagelevel import StorageLevel


class Indices:
    def __init__(self, feature_columns: List[str], label_col: str = 'cluster', centroid_cluster_col: str = 'cluster'):
        """
        Инициализация класса для расчета метрик кластеризации.
        :param feature_columns: список названий колонок с фичами
        :param label_col: название колонки с лейблами в основном датафрейме
        :param centroid_cluster_col: название колонки с номерами кластеров в датафрейме центроидов
        """
        self.feature_columns = feature_columns
        self.label_col = label_col
        self.centroid_cluster_col = centroid_cluster_col

    def compute_indices(self, df: DataFrame, 
                        centroids_df: DataFrame, 
                        num_clusters: Optional[int] = None,
                        sample_size: Optional[int] = None,
                        sample_frac: Optional[float] = None,
                        seed: int = 42) -> Dict[str, float]:
        """
        Расчет метрик кластеризации.
        
        :param df: датафрейм с фичами и лейблами
        :param centroids_df: датафрейм с центроидами кластеров
        :param num_clusters: количество кластеров (опционально)
        :param sample_size: количество объектов для семплирования из каждого кластера
        :param sample_frac: доля для семплирования из каждого кластера (0.0 - 1.0)
        :param seed: сид для воспроизводимости семплирования
        :return: словарь с метриками
        """
        # Определяем количество кластеров
        if num_clusters is None:
            self.num_clusters = centroids_df.select(self.centroid_cluster_col).distinct().count()
        else:
            self.num_clusters = num_clusters

        # Глобальное среднее
        self.global_means = df.select(*[F.avg(c).alias(c) for c in self.feature_columns]).first()
        
        # Размеры кластеров
        self.cluster_sizes = df.groupBy(self.label_col).agg(F.count('*').alias('size'))
        self.cluster_sizes.coalesce(1).persist(StorageLevel.MEMORY_ONLY).foreach(lambda x: x)
        
        # Количество объектов
        self.n = df.count()
        
        # WSS
        self.wss = self._calculate_wss(df, centroids_df)
        
        # BSS и Calinski-Harabasz
        self.bss, self.calinski = self._calculate_calinski_harabasz(df, centroids_df, self.num_clusters)
        
        # WB-index
        self.wb_index = self.wss / self.bss if self.bss != 0 else float('inf')
        
        # XU-index
        self.xu_index = (self.wss / (self.n ** 2)) * num_clusters
        
        # Silhouette Score
        sampled_df = self._smart_sampling(df, sample_size, sample_frac, seed) if sample_size or sample_frac else df
        self.silhouette = self._calculate_silhouette(sampled_df)

        self.cluster_sizes.unpersist()
        return {
            'Silhouette': self.silhouette,
            'WB-index': self.wb_index,
            'XU-index': self.xu_index,
            'Calinski-Harabasz': self.calinski,
            'WSS': self.wss
        }

    def _calculate_wss(self, df: DataFrame, centroids_df: DataFrame) -> float:
        """Вычисление Within-Cluster Sum of Squares (WSS)"""
        # Переименовываем колонки центроидов для объединения
        centroid_aliases = [F.col(c).alias(f'centroid_{c}') for c in self.feature_columns]
        centroids_renamed = centroids_df.select(
            *centroid_aliases, 
            F.col(self.centroid_cluster_col).alias('centroid_cluster')
        )
        
        # Объединяем данные с центроидами
        joined = df.join(
            centroids_renamed,
            df[self.label_col] == centroids_renamed['centroid_cluster']
        )
        
        # Вычисляем квадраты расстояний
        distance_expr = sum(
            F.pow(df[c] - centroids_renamed[f'centroid_{c}'], 2) 
            for c in self.feature_columns
        )
        
        return joined.select(F.sum(distance_expr)).first()[0]

    def _calculate_calinski_harabasz(self, df: DataFrame, centroids_df: DataFrame, k: int) -> (float, float):
        """Вычисление Calinski-Harabasz Index"""
        
        # Объединяем центроиды с размерами кластеров
        centroids_with_sizes = centroids_df.join(
            self.cluster_sizes,
            centroids_df[self.centroid_cluster_col] == self.cluster_sizes[self.label_col]
        )
        
        # BSS
        bss_expr = sum(
            F.pow(centroids_with_sizes[c] - self.global_means[c], 2) 
            for c in self.feature_columns
        )
        bss = centroids_with_sizes.select(
            F.sum(F.col('size') * bss_expr)
        ).first()[0]
        
        # Calinski-Harabasz
        if k == 1 or self.n == k:
            return bss, 0.0
        return bss, (bss / (k - 1)) / (self.wss / (self.n - k))

    def _smart_sampling(self, df: DataFrame, sample_size: int, sample_frac: float, seed: int) -> DataFrame:
        """Равномерное семплирование по кластерам"""
        cluster_counts = self.cluster_sizes.collect()

        # Определяем долю для семплирования
        fractions = {}
        for row in cluster_counts:
            cluster_id = row[self.label_col]
            count = row['size']
            
            if sample_size:
                frac = min(sample_size / count, 1.0)
            elif sample_frac:
                frac = sample_frac
            else:
                frac = 1.0
                
            fractions[cluster_id] = frac

        return df.sampleBy(self.label_col, fractions, seed)

    def _calculate_silhouette(self, df: DataFrame) -> float:
        """Расчет Silhouette Score через ClusteringEvaluator"""

        assembler = VectorAssembler(
            inputCols=self.feature_columns,
            outputCol="features_temp"
        )
        df_vec = assembler.transform(df).select("features_temp", self.label_col)
        
        # silhouette
        evaluator = ClusteringEvaluator(
            predictionCol=self.label_col,
            featuresCol="features_temp",
            metricName="silhouette"
        )
        return evaluator.evaluate(df_vec)