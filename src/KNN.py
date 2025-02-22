import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import Window
from pyspark.storagelevel import StorageLevel
import functools
import itertools
import concurrent
from typing import Iterator

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
        
        self._type_y = self.train_df.select(self._y_col).dtypes[0][1]
        assert self._type_y in ["double", 'float','int','bigint']

        self._feature_columns = self.train_df.drop(self._y_col).columns
        
        if self._type_y in ["double", 'float']:
            self._clf = False
        else:
            self._clf = True

    def fit(self, window_size = None, idcol:str = 'index_train') -> None:
        """
        Метод для предобработки "обучающего" датасета.
        window_size - количество строк в партиции. Если не задано, подсчет происходит автоматически.
        normalize - метод нормализации данных. Есть minmax, zscore или None.
        Нормализуются все столбцы кроме _idcol, _type_y. Следует передавать только фичи.
        """
        
        self._idcol = idcol          

        self.train_df_size = (self.train_df.count(), len(self._feature_columns)+2)
        self.train_df_window_size = int(((self.train_df_size[0] * self.train_df_size[1]) / (0.3 * 1024 * 1024)))
        
        kgroups_train = 1 if self.train_df_window_size <= 0 else self.train_df_window_size
        if not window_size == None:
            kgroups_train = int(self.train_df_size[0] // window_size)
        self.kgroups_train = 1 if kgroups_train <= 0 else kgroups_train

        if self._idcol not in self.train_df.columns:
            self.train_df = self.train_df.withColumn(self._idcol, F.row_number().over(Window.orderBy(F.rand())))
        
        self._feature_columns = self.train_df.drop(self._idcol, self._y_col).columns
        
        self.train_df = self.train_df.withColumn("rank", F.ntile(self.kgroups_train).over(Window.orderBy(self._idcol)))
        self.train_df = self.train_df.withColumn('features', F.array(*[col for col in self._feature_columns])).select('rank', self._idcol, 'features', self._y_col)
        self.train_df = self.train_df.repartitionByRange(self.kgroups_train, "rank").checkpoint()

        if self.train_df.select(self._idcol).distinct().count() != self.train_df.count():
            raise ValueError(f'Column {self._idcol} must contain unique values')

        self._train_chunks = self.train_df.select('rank').distinct().rdd.flatMap(lambda x: x).collect()
    
    
    def predict(self,
                pred_df: DataFrame, 
                pred_df_window_size: int = None,
                idcol:str = 'index_test',
                n_neighbors: int = 10, 
                metric: str = 'euclidean',
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
        self._idcol_test = idcol
        assert metric in ["cosine", 'euclidean','manhattan']

        self.pred_df_size = (pred_df.count(), len(self._feature_columns)+2)
        self.pred_df_window_size = int(((self.pred_df_size[0] * self.pred_df_size[1]) / (0.2 * 1024 * 1024)))
        kgroups_test = 1 if self.pred_df_window_size <= 0 else self.pred_df_window_size
        
        if not pred_df_window_size == None:
            kgroups_train = int(self.pred_df_size[0] // pred_df_window_size)
        self.kgroups_test = 1 if kgroups_test <= 0 else kgroups_test
        
        if self._idcol_test not in pred_df.columns:
            pred_df = pred_df.withColumn(self._idcol_test, F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))
        
        pred_df = pred_df.withColumn("rank", F.ntile(self.kgroups_test).over(Window.orderBy(self._idcol_test)))
        pred_df = pred_df.withColumn('features', F.array(*[col for col in pred_df.drop(self._idcol_test, 'rank').columns]))\
                            .select('rank', self._idcol_test, 'features')
        self.pred_df = pred_df.repartitionByRange(self.kgroups_test, "rank").checkpoint()

        if self.pred_df.select(self._idcol_test).distinct().count() != self.pred_df.count():
            raise ValueError(f'Column {self._idcol_test} must contain unique values')

        self._pred_chunks = self.pred_df.select('rank').distinct().rdd.flatMap(lambda x: x).collect()
        self.chunk_combinations = list(itertools.product(self._train_chunks, self._pred_chunks))

        
        def process_chunk_pair(pair, n_neighbors:int = n_neighbors, metric:str = 'cosine', weighted:bool = True):
            distance_functions = {
            'cosine': lambda x, y: float(1 - (np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))),
            'euclidean': lambda x, y: float(np.sqrt(np.dot(x-y, x-y))),
            'manhattan': lambda x, y: float(np.sum(np.abs(x - y)))}
    
            def calculate_distances(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
                dist_func = distance_functions[metric]
                for pdf in iterator:
                    pdf['distance'] = pdf.apply(lambda row: dist_func(
                        np.array(row['featuresA']), 
                        np.array(row['featuresB'])),axis=1)
                    yield pdf
            
            chunk_a, chunk_b = pair
            df_a = self.train_df.filter(F.col('rank') == chunk_a)#.hint('broadcast')
            df_b = self.pred_df.filter(F.col('rank') == chunk_b)#.hint('broadcast')
            #######################
            #df_a = F.broadcast(df_a)
            #df_b = F.broadcast(df_b)
            #######################
            
            result = df_a.alias('a').crossJoin(df_b.alias('b'))\
                        .select(
                            *[F.col(f'a.{self._idcol}').alias('index_train'), 
                            F.col(f'b.{self._idcol_test}').alias('index_test'), 
                            F.col(f'a.features').alias('featuresA'), 
                            F.col(f'b.features').alias('featuresB'), 
                            F.col(f'a.{self._y_col}').alias('y')])
            result = result.mapInPandas(calculate_distances, 
                                      result.withColumn('distance', F.lit(90.912591951925951)).schema)
            result = result.withColumn("row_number", F.rank().over(Window.partitionBy("index_test").orderBy("distance")))\
            .filter(F.col("row_number") <= n_neighbors)
            if weighted:
                result = result.withColumn("distance", F.when(F.col("distance") == 0.0, 1.0).otherwise(1 / F.col("distance")**2))
            result = result.select('index_train','index_test','distance','y').coalesce(1)
            result.persist(StorageLevel.MEMORY_ONLY).foreach(lambda x: x)
            return result

        
        results = []
        errors = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
            combinations = {executor.submit(process_chunk_pair, chunk, n_neighbors, metric, weighted): chunk for chunk in self.chunk_combinations}
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
            with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
                combinations = {executor.submit(process_chunk_pair, chunk, n_neighbors, metric, weighted): chunk for chunk in self.chunk_combinations}
                for future in concurrent.futures.as_completed(combinations):
                    chunk = combinations[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        print('%r generated as exception %s' % (chunk, exc))
                        errors.append(chunk)

        similarity = functools.reduce(lambda a,b: a.union(b), results)
        self.similarity = similarity.repartitionByRange(self.kgroups_test, 'index_test').checkpoint()
        [i.unpersist() for i in results]
        
        
        self.similarity = self.similarity.withColumn("row_number", F.rank().over(Window.partitionBy("index_test").orderBy("distance")))\
                                        .filter(F.col("row_number") <= n_neighbors).drop("row_number")
        if self._clf:
            if weighted:
                self.similarity = self.similarity.groupBy('index_test','y').agg(F.sum('distance').alias('metric'))
            else:
                self.similarity = self.similarity.groupBy('index_test','y').agg(F.count('y').alias('metric'))
            self.similarity = self.similarity.withColumn("row_number", F.row_number().over(Window.partitionBy("index_test").orderBy(F.col('metric').desc())))\
                            .filter(F.col("row_number") == 1).drop("row_number")
        else:
            if weighted:
                self.similarity = self.similarity.groupBy("index_test").agg((F.sum(F.col("y") * F.col("distance")) / F.sum(F.col("distance"))).alias("y"))
            else:
                self.similarity = self.similarity.groupBy('index_test').agg(F.avg('y').alias('y'))
        self.similarity = self.similarity.withColumnRenamed('y', self._y_col).withColumnRenamed('index_test', self._idcol_test).checkpoint()
        return self.similarity

