from graphframes import GraphFrame
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.storagelevel import StorageLevel
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import Window
import functools
import itertools
import concurrent
from typing import Iterator, Dict

class HiClust:
    """
    Кастомная иерархическая кластеризация
    """
    def __init__(self,
                 spark: SparkSession,
                 train_df: DataFrame,
                 window_size = None,
                 idcol:str = 'index_train'
                ) -> None:
        """
        Предобработка датасета для кластеризации.
        window_size - количество строк в партиции. Если не задано, подсчет происходит автоматически.
        Желательно нормализовать все столбцы кроме idcol. 
        В объект следует передавать только фичи, по которым должна происходить кластеризация.
        """
        self.spark = spark
        self.train_df = train_df
        self._idcol = idcol
        self.cluster_history = None
        self.dict_iteration = {}
        self.dict_centroids_iteration = {}
        if self._idcol not in self.train_df.columns:
            self.train_df = self.train_df.withColumn(self._idcol, F.row_number().over(Window.orderBy(F.rand())))
        
        self._feature_columns = self.train_df.drop(self._idcol).columns
        
        self.train_df_size = (self.train_df.count(), len(self._feature_columns)+2)
        self.train_df_window_size = int(((self.train_df_size[0] * self.train_df_size[1]) / (0.2 * 1024 * 1024)))
        
        kgroups_train = 1 if self.train_df_window_size <= 0 else self.train_df_window_size
        if not window_size == None:
            kgroups_train = int(self.train_df_size[0] // window_size)
        self.kgroups_train = 1 if kgroups_train <= 0 else kgroups_train
        
        self.train_df = self.train_df.withColumn("rank", F.ntile(self.kgroups_train).over(Window.orderBy(self._idcol)))
        self.train_df = self.train_df.withColumn('features', F.array(*[col for col in self._feature_columns])).select('rank', self._idcol, 'features')
        self.train_df = self.train_df.repartitionByRange(self.kgroups_train, "rank").checkpoint()
        
        if self.train_df.select(self._idcol).distinct().count() != self.train_df.count():
            raise ValueError(f'Column {self._idcol} must contain unique values')

        self.dict_iteration[0] = self.train_df.withColumn('cluster_id_0', F.monotonically_increasing_id())\
                            .select(self._idcol, 'cluster_id_0').checkpoint(False)
        self.dict_centroids_iteration[0] = None

        self._train_chunks = self.train_df.select('rank').distinct().rdd.flatMap(lambda x: x).collect()
        self.chunk_combinations = list(itertools.product(self._train_chunks, repeat=2))
        self.chunk_combinations = [i for i in self.chunk_combinations if i[0] <= i[1]]
        return None

    
    def fit(self, metric: str = 'euclidean', distance_threshold: float = 0.1, max_iterations:int = 10, auto_threshold: bool = False, find_top_one:bool = False) -> None:
        
        assert metric in ["cosine", 'euclidean','manhattan']

        iteration_i = 0
        count_clusters = float('inf')

        if metric == 'cosine':
            self.train_df = self._l2_norm_df(self.train_df).checkpoint(False)
            
        train_df = self.train_df.withColumn('dist_weight', F.lit(1.0))

        try:
            while iteration_i < max_iterations:
                # Создание матрицы схожести

                if auto_threshold:
                    result = self._create_sim_matrix(df=train_df, metric=metric,
                                               distance_threshold=float('inf'))
                    dist_stats = result.agg(F.mean('distance').alias('avg(distance)'),
                                            F.stddev('distance').alias('std(distance)')).first()
                    print(f'iter:{iteration_i+1} ',dist_stats['avg(distance)'],dist_stats['std(distance)'])
                    if dist_stats['std(distance)'] is not None:
                        newthreshold = dist_stats['avg(distance)'] + 3 * dist_stats['std(distance)']
                        distance_threshold = newthreshold if (distance_threshold > newthreshold) else distance_threshold
                        result = result.filter(F.col('distance') <= distance_threshold)
                        print(f'distance_threshold:{distance_threshold}')
                else:
                    result = self._create_sim_matrix(df=train_df, metric=metric,
                                               distance_threshold=distance_threshold)
                
                if find_top_one:
                    result = self._find_top_one(result)
                    
                if result.rdd.count() == 0:
                    print(f"All distances exceed threshold {distance_threshold}")
                    break
        
                # Обработка соседей
                edges = result.select(F.col(f'{self._idcol}_A').alias("src"), 
                                      F.col(f'{self._idcol}_B').alias("dst"))
                
                if edges.rdd.count() == 0:
                    print("No edges to process")
                    break
                    
                # Поиск связанных компонентов
                all_nodes = train_df.select(F.col(self._idcol)).distinct()
                vertices = all_nodes.withColumnRenamed(self._idcol, "id")
    
                G = GraphFrame(vertices, edges)
                cc = G.connectedComponents(algorithm='graphx', checkpointInterval=1, broadcastThreshold=104857600)
                
                # Обновление маппинга кластеров
                iteration_i += 1
                cluster_col = f'cluster_id_{iteration_i}'
                iteration = cc.select(F.col("id").alias(self._idcol), F.col("component").alias(cluster_col)).checkpoint()
                
                self.dict_iteration[iteration_i] = iteration
                new_count = iteration.select(cluster_col).distinct().count()
                print(f'iteration: {iteration_i}, count clusters = {new_count}')
                
                # Проверка условия останова
                if new_count >= count_clusters:
                    print(f"Clusters stabilized at {count_clusters}")
                    break
                    
                count_clusters = new_count
                
                # Пересчет центроидов
                train_df = self._calculate_centroids(
                    train_df.join(iteration, self._idcol, "left"),
                    cluster_col, metric).checkpoint()
                self.dict_centroids_iteration[iteration_i] = train_df
                
                if 'rank' not in train_df.columns:
                    self.train_df_size = (train_df.count(), len(self._feature_columns)+2)
                    self.train_df_window_size = int(((self.train_df_size[0] * self.train_df_size[1]) / (0.2 * 1024 * 1024)))
                    kgroups_train = 1 if self.train_df_window_size <= 0 else self.train_df_window_size
                    self.kgroups_train = 1 if kgroups_train <= 0 else kgroups_train
                    train_df = train_df.withColumn("rank", F.ntile(self.kgroups_train).over(Window.orderBy(F.col(f'cluster_id_{iteration_i}'))))
                    train_df = train_df.withColumn('features', F.array(*[col for col in self._feature_columns]))\
                                        .select('rank', F.col(f'cluster_id_{iteration_i}').alias(self._idcol), 'features','dist_weight')
                    train_df = train_df.repartitionByRange(self.kgroups_train, "rank").checkpoint()
                    self._train_chunks = train_df.select('rank').distinct().rdd.flatMap(lambda x: x).collect()
                    self.chunk_combinations = list(itertools.product(self._train_chunks, repeat=2))
                    self.chunk_combinations = [i for i in self.chunk_combinations if i[0] <= i[1]]
        except Exception as e:
            print(f"Clustering failed: {str(e)}")
        finally:
            # Финализация истории кластеров
            self.cluster_history = self.combine_cluster_iterations()
        return None

    def _find_top_one(self, df: DataFrame) -> DataFrame:
        window = Window.partitionBy(f'{self._idcol}_A').orderBy('distance')
        df = df.withColumn('rn', F.row_number().over(window))\
                          .filter(F.col('rn') == 1)\
                          .drop('rn')
        
        # Фильтрация взаимных пар
        df = df.alias('n1').join(
            df.alias('n2'),
            (F.col(f'n1.{self._idcol}_A') == F.col(f'n2.{self._idcol}_B')) &
            (F.col(f'n1.{self._idcol}_B') == F.col(f'n2.{self._idcol}_A'))
        ).select(
            F.col(f'n1.{self._idcol}_A').alias(f'{self._idcol}_A'),
            F.col(f'n1.{self._idcol}_B').alias(f'{self._idcol}_B'),
            F.col('n1.distance').alias('distance')
        )
        return df

    def _l2_norm_df(self, df: DataFrame) -> DataFrame:
        normalized_df = df.withColumn("features",
                F.transform("features", lambda x: x / F.sqrt(F.aggregate(
                F.transform("features", lambda y: y ** 2), F.lit(0.0), lambda acc, y: acc + y))))
        return normalized_df

    def _calculate_centroids(self, df: DataFrame, cluster_col: str, metric: str) -> DataFrame:
        """Расчет центроидов с учетом метрики"""
        agg_exprs = [F.avg(F.col('features').getItem(i)).alias(c) for i, c in enumerate(self._feature_columns)]
        centroids = df.groupBy(cluster_col).agg(*agg_exprs, F.sum('dist_weight').alias('dist_weight'))
        return centroids

    
    def combine_cluster_iterations(self) -> DataFrame:
        """Объединение истории кластеризации"""
        sorted_iters = sorted(self.dict_iteration.keys())
        
        if not sorted_iters:
            raise ValueError("No clustering iterations available")
        
        result_df = self.dict_iteration[0]
        
        for iter_num in sorted_iters[1:]:
            if iter_num not in self.dict_iteration:
                raise KeyError(f"Missing iteration {iter_num} in history")
            if (iter_num - 1) not in self.dict_iteration:
                raise KeyError(f"Previous iteration {iter_num-1} not found")
            prev_iter = iter_num - 1
            prev_col = f"cluster_id_{prev_iter}"
            curr_col = f"cluster_id_{iter_num}"
            
            max_id = result_df.agg(F.max(prev_col)).first()[0] or 0
            current_df = self.dict_iteration[iter_num].withColumn(
                curr_col, F.col(curr_col) + max_id + 1
            )
            
            result_df = result_df.join(
                current_df,
                on=self._idcol,
                how="left"
            ).withColumn(
                curr_col,
                F.coalesce(F.col(curr_col), F.col(prev_col))
            )
        
        return result_df
    
    def get_clusters(self, iteration: int = None) -> DataFrame:
        """Получение кластеров для конкретной итерации"""
        if not self.cluster_history:
            raise ValueError("Model not fitted yet")
            
        if iteration is None:
            iteration = max(self.dict_iteration.keys())
            
        cluster_col = f"cluster_id_{iteration}"
        
        return self.cluster_history.select(self._idcol, cluster_col)

    
    def _create_sim_matrix(self, df: DataFrame, metric:str, distance_threshold: float) -> DataFrame:
        """Расчет матрицы схожести с выбранной метрикой."""
        distance_functions = {
        'cosine': lambda x, y: float(1 - (np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))),
        'euclidean': lambda x, y: float(np.sqrt(np.dot(x-y, x-y))),
        'manhattan': lambda x, y: float(np.sum(np.abs(x - y)))}

        def calculate_distances(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
            dist_func = distance_functions[metric]
            for pdf in iterator:
                pdf['distance'] = pdf.apply(
                    lambda row: dist_func(
                        np.array(row['featuresA']), 
                        np.array(row['featuresB'])
                    ) * float(row['dst_weight']),
                    axis=1
                )
                yield pdf[pdf['distance'] <= distance_threshold]
                
        def process_chunk_pair(pair, metric):
            chunk_a, chunk_b = pair
            df_a = df.filter(F.col('rank') == chunk_a)#.hint('broadcast')
            df_b = df.filter(F.col('rank') == chunk_b)#.hint('broadcast')
            #######################
            #df_a = F.broadcast(df_a)
            #df_b = F.broadcast(df_b)
            #######################
            
            #.withColumn('dst_weight', F.sqrt(F.col(f'a.dist_weight') * F.col(f'b.dist_weight')))\
            
            result = df_a.alias('a').crossJoin(df_b.alias('b'))\
                        .filter(F.col(f'a.{self._idcol}') < F.col(f'b.{self._idcol}'))\
                        .withColumn('dst_weight', 
                                    (2* F.least(F.col(f'a.dist_weight'), F.col(f'b.dist_weight'))) /
                                    (F.col(f'a.dist_weight') + F.col(f'b.dist_weight')))\
                        .select(
                            *[F.col(f'a.{self._idcol}').alias(f'{self._idcol}_A'),
                            F.col(f'b.{self._idcol}').alias(f'{self._idcol}_B'),
                            F.col(f'a.features').alias(f'featuresA'),
                            F.col(f'b.features').alias(f'featuresB'),
                            F.col(f'dst_weight')]
                        )
            result = result.mapInPandas(calculate_distances, 
                                      result.withColumn('distance', F.lit(90.912591951925951)).schema)
            result = result.select(F.col(f'{self._idcol}_A'),F.col(f'{self._idcol}_B'),'distance').coalesce(1)
            result.persist(StorageLevel.DISK_ONLY).foreach(lambda x: x)
            return result
        
        results = []
        errors = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            combinations = {executor.submit(process_chunk_pair, chunk, metric): chunk for chunk in self.chunk_combinations}
            for future in concurrent.futures.as_completed(combinations):
                chunk = combinations[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    print('%r generated as exception %s' % (chunk, exc))
                    errors.append(chunk)
                    
        while len(errors) > 0:
            print(f'generated {len(errors)} errors')
            self.chunk_combinations = errors
            errors = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
                combinations = {executor.submit(process_chunk_pair, chunk, metric): chunk for chunk in self.chunk_combinations}
                for future in concurrent.futures.as_completed(combinations):
                    chunk = combinations[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        print('%r generated as exception %s' % (chunk, exc))
                        errors.append(chunk)

        similarity = functools.reduce(lambda a,b: a.union(b), results)
        self.sim_matrix = similarity.repartitionByRange(self.kgroups_train, f'{self._idcol}_A').checkpoint()
        [i.unpersist() for i in results]
        return self.sim_matrix