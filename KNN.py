import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import Window
from sklearn.neighbors import NearestNeighbors
import functools
import itertools


class spark_knn:
    """
    Кастомный класс для KNN в pyspark
    """
    def __init__(self,
                 spark: SparkSession,
                 train_df: DataFrame,
                 y_col: str = 'y') -> None:
        """
        spark - спарк сессия
        train_df - "обучающий" датасет
        y_col - названия столбца с зависимой переменной. Если не задано, то - y
        """
        
        self.spark = spark
        self.train_df = train_df
        self.pred_df = None
        
        if not y_col in self.train_df.columns:
            raise AttributeError(f'Invalid column: {y_col}')
        self._y_col = y_col
        self.train_df.withColumnRenamed(self._y_col, 'y')
        
        self._type_y = self.train_df.select('y').dtypes[0][1]
        assert self._type_y in ["double", 'float','int','bigint']

        self._feature_columns = self.train_df.drop('y').columns
        
        if self._type_y in ["double", 'float']:
            self._clf = False
        else:
            self._clf = True
            
        self._euclidean_distances_udf = F.udf(spark_knn._euclidean_distances, returnType=T.DoubleType())
        self._manhattan_distances_udf = F.udf(spark_knn._manhattan_distances, returnType=T.DoubleType())
        self._cosine_distance_udf = F.udf(spark_knn._cosine_distance, returnType=T.DoubleType())
        self._squared_euclidean_distance = F.udf(spark_knn._squared_euclidean_distance, returnType=T.DoubleType())

    def fit(self, window_size = None) -> None:
        """
        Метод для предобработки "обучающего" датасета.
        window_size - количество строк в партиции. Если не задано, подсчет происходит автоматически.
        Нормализация внутри алгоритма не предусмотрена.
        """
        train_df_size = (self.train_df.count(), len(self._feature_columns)+2)
        train_df_window_size = int(((train_df_size[0] * train_df_size[1]) / (6 * 1024 * 1024)))
        kgroups_train = 1 if train_df_window_size <= 0 else train_df_window_size
        if not window_size == None:
            kgroups_train = int(train_df_size[0] // window_size)
        self.kgroups_train = 1 if kgroups_train <= 0 else kgroups_train
        if 'index_train' not in self.train_df.columns:
            self.train_df = self.train_df.withColumn("index_train", F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))
        self.train_df = self.train_df.withColumn("rank", F.ntile(self.kgroups_train).over(Window.orderBy("index_train")))
        self.train_df = self.train_df.repartition(self.kgroups_train, "rank").checkpoint()

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
    
    
    def predict(self,
                pred_df: DataFrame, 
                pred_df_window_size: int = None, 
                n_neighbors: int = 10, 
                metric: str = 'euclidean',
                k_iter_union: int = 50,
                weighted: bool = True) -> DataFrame:
        """
        Метод для предсказания по "обучающему датасету".
        pred_df - датасет идентичный train_df без столбца с зависимой переменной.
        pred_df_window_size - количество строк в партиции. Если не задано, подсчет происходит автоматически.
        n_neighbors - количество ближайших соседей.
        metric - метрика для дистанции ["cosine", 'euclidean','manhattan']
        weighted - использовать взвешенный KNN
        k_iter_union - Количество партиций для объединения. Следует регулировать при больших датасетах.
        """
        
        assert metric in ["cosine", 'euclidean','manhattan']
        
        def funcKNN(data: pd.DataFrame) -> pd.DataFrame:
            """
            Метод для поиска ближайшего соседа и фильтрации по трешхолду
            """
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm='brute', n_jobs=-1)
            nn.fit(data.drop(['index_train','rank','y'], axis=1))
            distances, indices = nn.kneighbors(testk.value.drop(['index_test'], axis=1))
            df_result = pd.DataFrame({
            'index_test': testk.value['index_test'].repeat(n_neighbors),
            'index_train': data.reset_index().iloc[indices.flatten()]['index_train'].tolist(),
            'distance': distances.flatten(),
            'y': data.reset_index().iloc[indices.flatten()]['y'].tolist()
            })
            return df_result

        def _union_list_ch(union_ch: list, list_ch_k: list):
            """
            Метод для объединения чанков в более крупные чанки
            """
            unioned_ch = functools.reduce(lambda x, y: x.union(y), list_ch_k)
            unioned_ch = unioned_ch.checkpoint()
            union_ch.append(unioned_ch)
            list_ch_k.clear()
            return None

        def _calculate(df: DataFrame, clf: bool, weighted: bool, n_neighbors: int) -> DataFrame:
            """
            Фильтрация по ближайшим соседям
            """
            df = df.withColumn("row_number", F.rank().over(Window.partitionBy("index_test").orderBy("distance"))).filter(F.col("row_number") <= n_neighbors)
            if self._clf:
                if weighted:
                    df = df.withColumn("weight", F.when(F.col("distance") == 0.0, 1.0).otherwise(1 / F.col("distance")**2))
                    df = df.groupBy('index_test','y').agg(F.sum('weight').alias('metric'))
                else:
                    df = df.groupBy('index_test','y').agg(F.count('y').alias('metric'))
                df = df.withColumn("row_number", F.row_number().over(Window.partitionBy("index_test").orderBy(F.col('metric').desc())))\
                                .filter(F.col("row_number") == 1).drop("row_number")
            else:
                if weighted:
                    df = df.withColumn("weight", F.when(F.col("distance") == 0.0, 1.0).otherwise(1 / F.col("distance")**2))
                    df = df.groupBy("index_test").agg((F.sum(F.col("y") * F.col("weight")) / F.sum(F.col("weight"))).alias("y"))
                else:
                    df = df.groupBy('index_test').agg(F.avg('y').alias('y'))
            df = df.withColumnRenamed('y', self._y_col).coalesce(1).checkpoint()
            return df
        
        if pred_df_window_size == None:
            pred_df_size = (pred_df.count(), len(self._feature_columns)+1)
            pred_df_window_size = int((pred_df_size[0] * pred_df_size[1]) / (1024 * 1024))
            kgroups_test = 1 if pred_df_window_size <= 0 else pred_df_window_size
        elif (pred_df_window_size > 0) and isinstance(pred_df_window_size, int):
            kgroups_test = int(pred_df.count() // pred_df_window_size)
            kgroups_test = 1 if kgroups_test <= 0 else kgroups_test
        else:
            raise ValueError('pred_df_window_size must be greater None or int value')

        if 'index_test' not in pred_df.columns:
            pred_df = pred_df.withColumn("index_test", F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))
        self.pred_df = pred_df.withColumn("rank", F.ntile(kgroups_test).over(Window.orderBy("index_test"))).repartition(kgroups_test, "rank").checkpoint()
    
        if kgroups_test == 1:
            testk = self.spark.sparkContext.broadcast(self.pred_df.drop('rank').toPandas())
            _result = self.train_df.groupBy('rank').applyInPandas(funcKNN, schema=f'index_test: bigint, index_train:bigint, distance: float, y: {self._type_y}')
            _result = _calculate(_result, self._clf, weighted, n_neighbors)
            testk.destroy()
        else:
            list_ch_k = []
            union_ch = []
            k_iter = 0
            for i in range(0, kgroups_test):
                testk = self.spark.sparkContext.broadcast(self.pred_df.filter(F.col('rank') == i+1).drop('rank').toPandas())
                _result = self.train_df.groupBy('rank').applyInPandas(funcKNN, schema=f'index_test: bigint, index_train:bigint, distance: float, y: {self._type_y}')
                _result = _calculate(_result, self._clf, weighted, n_neighbors)
                testk.destroy()
                list_ch_k.append(_result)
                if k_iter % k_iter_union == 0:
                    _union_list_ch(union_ch, list_ch_k)
            if len(list_ch_k) > 0:
                _union_list_ch(union_ch, list_ch_k)
            if len(union_ch) == 1:
                _result = union_ch[0]
            else:
                _result = functools.reduce(lambda x, y: x.union(y), union_ch)
        return _result

