from graphframes import GraphFrame
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import Window
from sklearn.neighbors import NearestNeighbors
import functools
import itertools


class HiClust:
    """
    Кастомная иерархическая кластеризация
    
    1) Считаем дистанцию между центроидами.
    2) Возьмем только топ 1 центроид по заданной метрике, взвешивая дистанцию на инерцию центроидов.
    3) Оставляем только те, что проходят по трешхолду
    4) Объединяем пары в граф без конфликтов.
    5) Считаем центроид графа и инерцию.
    6) Повторяем с пункта 1 до тех пор, пока количетсво кластеров за итерацию не перестанет уменьшаться. 

    spark - спарк сессия
    train_df - датасет для кластеризации. Нужен столбец с индексами с названием index_train. Если не задан, создается автоматически.
    """
    def __init__(self,
                 spark: SparkSession,
                 train_df: DataFrame) -> None:
        
        self.spark = spark
        self.train_df = train_df

        if 'index_train' not in train_df.columns:
            self.train_df = self.train_df.withColumn("index_train", F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))
        self._feature_columns = self.train_df.drop('index_train').columns
    
    def fit(self, window_size = None, normalize: str = None) -> None:
        """
        Метод для предобработки датасета для кластеризации
        window_size - количество строк в партиции. Если не задано, подсчет происходит автоматически.
        normalize - метод нормализации данных. Есть minmax, zscore или None.
        Нормализуются все столбцы кроме index_train. Следует передавать только фичи, по которым должна происходить кластеризация.
        """
        assert normalize in ["minmax", 'zscore', None]
        if normalize == 'zscore':
            for col in self._feature_columns:
                mean_val = self.train_df.agg({col: "mean"}).first()[0]
                std_val = self.train_df.agg({col: "stddev"}).first()[0]
                self.train_df = self.train_df.withColumn(col, (F.col(col) - mean_val) / std_val)
            print('zscore normalized')
        elif normalize == 'minmax':
            for col in self._feature_columns:
                min_val = self.train_df.agg({col: "min"}).first()[0]
                max_val = self.train_df.agg({col: "max"}).first()[0]
                self.train_df = self.train_df.withColumn(col, (F.col(col) - min_val) / (max_val - min_val))
            print('minmax normalized')
        else:
            print('df is not normalized - continue')
        
        max_index_train = self.train_df.select(F.max("index_train")).collect()[0][0]
        train_df_size = (max_index_train, len(self._feature_columns)+2)
        train_df_window_size = int(((train_df_size[0] * train_df_size[1]) / (6 * 1024 * 1024)))
        kgroups_train = 1 if train_df_window_size <= 0 else train_df_window_size
        if not window_size == None:
            kgroups_train = int(train_df_size[0] // window_size)
        self.kgroups_train = 1 if kgroups_train <= 0 else kgroups_train
        if 'index_train' not in self.train_df.columns:
            self.train_df = self.train_df.withColumn("index_train", F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))
        self.train_df = self.train_df.withColumn("rank", F.ntile(self.kgroups_train).over(Window.orderBy("index_train")))
        self.train_df = self.train_df.repartitionByRange(self.kgroups_train, "rank").checkpoint()
        return None

    @staticmethod
    def _euclidean_distances(x: list, y: list) -> float:
        x = np.array(x)
        y = np.array(y)
        return float(np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y)))
    
    @staticmethod
    def _squared_euclidean_distance(x: list, y: list) -> float:
        return float(np.sum((np.array(x) - np.array(y)) ** 2))
    
    @staticmethod
    def _manhattan_distances(x: list, y: list) -> float:
        return float(sum(abs(a - b) for a, b in zip(x, y)))
    
    @staticmethod
    def _cosine_distance(x: list, y: list) -> float:
        x = np.array(x)
        y = np.array(y)
        return float(1 - (np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))))

    def _historyPredict(self) -> None:
        """
        Метод для объединения этапов кластеризации в единую цепочку. Форматирует output алгоритма.
        """
        self.hist = self.dict_iteration.get(0)[1].select(F.col('index_train'), F.col('cluster_id').alias('cluster_id_1'))
        for i in range(1, len(self.dict_iteration)):
            self.hist = self.hist.join(self.dict_iteration.get(i)[1].select(F.col('index_train').alias(f'cluster_id_{i}'), F.col(f'cluster_id').alias(f'cluster_id_{i+1}')), on=f'cluster_id_{i}', how='left')
        self.hist = self.hist.checkpoint()
        return None
    
    
    def predict(self,
                metric: str = 'euclidean',
                k_iter_union: int = 50,
                distance_threshold: float = 0.1,
                weighted_first_iter: bool = False) -> DataFrame:
        """
        Метод для поиска кластеров в датасете
        metric - метрика дистанции ["cosine", 'euclidean','manhattan', 'seuclidean']
        k_iter_union - Количество партиций для объединения. Следует регулировать при больших датасетах.
        distance_threshold - Трешхолд в кластеризации.
        weighted_first_iter - Необходимо ли взвешивать на первой итерации дистанцию между синглтонами.
        """
        
        assert metric in ["cosine", 'euclidean','manhattan', 'seuclidean']

        _functions_dist = {
        'euclidean': self._euclidean_distances,
        'cosine': self._manhattan_distances,
        'manhattan': self._cosine_distance,
        'seuclidean': self._squared_euclidean_distance
        }
        function_distance = _functions_dist.get(metric)
        self.dict_iteration = {}
        

        def searchNeighbors(df: DataFrame, flag: bool) -> DataFrame:
            """
            Метод для поиска ближайших соседей
            """
            
            def custom_distance(x, y):
                """
                Обертка над методом поиска дистанции
                """
                dist = function_distance(x[:-1], y[:-1])
                return ((x[-1] + y[-1])/2) * dist
            
            def _union_list_ch(union_ch: list, list_ch_k: list):
                """
                Метод для объединения чанков в более крупные чанки
                """
                unioned_ch = functools.reduce(lambda x, y: x.union(y), list_ch_k)
                unioned_ch = unioned_ch.checkpoint()
                union_ch.append(unioned_ch)
                list_ch_k.clear()
                return None
                
            def funcKNN(data: pd.DataFrame) -> pd.DataFrame:
                """
                Метод для поиска ближайшего соседа и фильтрации по трешхолду
                """
                n_neighbors = 1 if data.shape[0] <= 2 else 3
                if ('count' in data.columns) or ('mean' in data.columns):
                    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=custom_distance, algorithm='brute', n_jobs=-1)
                else:
                    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=function_distance, algorithm='brute', n_jobs=-1)
                nn = nn.fit(data.drop(['index_train','rank'], axis=1))
                distances, indices = nn.kneighbors(testk.value.drop(['index_test'], axis=1), n_neighbors=n_neighbors, return_distance=True)
                df_result = pd.DataFrame({
                    'index_test': testk.value['index_test'].repeat(n_neighbors),
                    'index_train': data.reset_index().iloc[indices.flatten()]['index_train'].tolist(),
                    'distance': distances.flatten()})
                df_result = df_result.loc[(df_result.index_test != df_result.index_train)].reset_index(drop=True)
                df_result = df_result.loc[df_result.groupby('index_test')['distance'].idxmin()]
                if ('count' in data.columns) or ('mean' in data.columns):
                    df_result = df_result.loc[df_result.distance <= distance_threshold]
                return df_result
            
            list_ch_k = []
            union_ch = []
            k_iter = 0
            for i in range(0, df.select('rank').distinct().count()):
                testk = df.withColumnRenamed('index_train','index_test').filter(F.col('rank') == i+1).drop('rank')
                if flag:
                    testk = testk.withColumn('count', F.lit(1))
                    _result = df.withColumn('count', F.lit(1))
                else:
                    _result = df
                testk = self.spark.sparkContext.broadcast(testk.toPandas())
                _result = _result.groupBy('rank').applyInPandas(funcKNN, schema=f'index_test: bigint, index_train:bigint, distance: float').coalesce(1).checkpoint()
                testk.destroy()
                list_ch_k.append(_result)
                if k_iter % k_iter_union == 0:
                    _union_list_ch(union_ch, list_ch_k)
            if len(list_ch_k) > 0:
                _union_list_ch(union_ch, list_ch_k)
            _result = functools.reduce(lambda x, y: x.union(y), union_ch)
            resultNeighbors = _result.withColumn("row_num", F.row_number().over(Window.partitionBy("index_test").orderBy(F.col("distance").asc())))\
                            .filter(F.col("row_num") == 1).drop("row_num")\
                            .groupBy("index_train").agg(F.collect_set("index_test").alias("group")).checkpoint()
            return resultNeighbors
        
        def graphiteration(resultNeighbors: DataFrame) -> DataFrame:
            """
            Метод для построения графа при отсутствии конфликтов
            """
            edges = resultNeighbors.select(F.col('index_train').alias("src"), F.explode('group').alias("dst"))
            vertices = resultNeighbors.select(F.col('index_train').alias("id"))
            G = GraphFrame(vertices, edges)
            cc = G.connectedComponents(algorithm='graphx', checkpointInterval=1, broadcastThreshold=104857600)
            iteration = resultNeighbors.join(cc.select(F.col('id').alias('index_train'), F.col('component').alias('cluster_id')), on='index_train', how='left')\
                                .drop('index_train')\
                                .select('cluster_id',F.explode('group').alias('index_train')).checkpoint()
            return iteration

        def calculateCentroids(df: DataFrame, iteration: DataFrame, feature_columns:list) -> DataFrame:
            """
            Поиск центроидов после итерации кластеризации
            """
            function_distance_udf = F.udf(function_distance, returnType=T.FloatType())
            
            train_df_with_clusters =  df.join(iteration, on="index_train", how="left")\
                                        .withColumn("cluster_id", F.coalesce('cluster_id','index_train'))
            
            maping_cluster = train_df_with_clusters.select('index_train','cluster_id').distinct().checkpoint()
            
            centroid_iteration = train_df_with_clusters.groupBy('cluster_id')\
                                    .agg(*[F.avg(i).alias(f"centroid_{i}") for i in feature_columns])
            centroid_cols = centroid_iteration.drop('index_train', 'count','cluster_id').columns
            
            train_df_with_clusters = train_df_with_clusters.join(centroid_iteration, on='cluster_id', how='left')\
                            .select('cluster_id', function_distance_udf(F.array(feature_columns), F.array(centroid_cols)).alias('distance'), *centroid_cols)\
                            .groupBy('cluster_id').agg(F.avg('distance').alias('mean'))
            centroid_iteration = centroid_iteration.join(train_df_with_clusters, on ='cluster_id', how='left')\
                                                    .withColumn("mean", F.when(F.col('mean') == 0.0, 1.0).otherwise(F.col('mean'))).checkpoint()
            return centroid_iteration, maping_cluster

        iteration_i = 0
        train_df = self.train_df
        count_clusters = float('inf')
        while True:
            try:
                if weighted_first_iter and (iteration_i == 0):
                    resultNeighbors = searchNeighbors(train_df, flag = True)
                else:
                    resultNeighbors = searchNeighbors(train_df, flag = False)
            except:
                if train_df.count() < 3:
                    self._historyPredict()
                    break
                print('coalesce all')
                train_df = train_df.withColumn("rank", F.ntile(1).over(Window.orderBy("index_train")))
                resultNeighbors = searchNeighbors(train_df, flag = False)
            iteration = graphiteration(resultNeighbors)
            centroid_iteration, maping_cluster = calculateCentroids(df = train_df, iteration = iteration, feature_columns= self._feature_columns)
            count_clusters_iter = centroid_iteration.select('cluster_id').distinct().count()
            print(f'iter {iteration_i+1} get {count_clusters_iter} clusters')
            if count_clusters == count_clusters_iter:
                print(f'end... Neither neighbor overcame the trashhold {distance_threshold}', count_clusters, count_clusters_iter)
                self._historyPredict()
                break
            count_clusters = count_clusters_iter
            self.dict_iteration[iteration_i] = [centroid_iteration, maping_cluster]
            iteration_i += 1
            train_df = centroid_iteration.withColumn("rank", F.ntile(self.kgroups_train).over(Window.orderBy("cluster_id")))
            train_df = train_df.toDF(*['index_train', *self._feature_columns, 'mean', 'rank'])
            
            max_index_train = train_df.select(F.max("index_train")).collect()[0][0]
            train_df_size = (max_index_train, len(self._feature_columns))
            train_df_window_size = int(((train_df_size[0] * train_df_size[1]) / (3 * 1024 * 1024)))
            kgroups_train = 1 if train_df_window_size <= 0 else train_df_window_size
            train_df = train_df.withColumn("rank", F.ntile(kgroups_train).over(Window.orderBy("index_train")))
            train_df = train_df.repartitionByRange(kgroups_train, "rank").checkpoint()
        return None

