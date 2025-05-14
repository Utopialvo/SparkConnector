import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import Window
from typing import Iterator
from pyspark.storagelevel import StorageLevel
import functools
import itertools
import concurrent


class GeneratorPysparkDf():
    """
    Генератор батчей для итеративной обработки данных в Spark.
    
    Пример использования:
    >>> spark = SparkSession.builder.getOrCreate()
    >>> gen = GeneratorPysparkDf(spark, "features_table", "adj_matrix_table")
    >>> for epoch, batch_df, adj_df in gen.get_df_generator(quadratic=True):
    ...     process_batch(batch_df, adj_df)
    
    Args:
        spark (SparkSession): Сессия Spark
        df (str): Название таблицы с фичами
        df_adj (str, optional): Название таблицы с матрицей смежности
    """
    def __init__(self, spark: SparkSession, df:str, df_adj:str = None):
        self.spark = spark
        self.splited_data = None
        self.kgroups = None
        self.dataframe = self.spark.read.table(f'{df}')
        self.adjdataframe = None
        self.count_df = self.dataframe.count()
        if df_adj != None:
            self.adjdataframe = self.spark.read.table(f'{df_adj}')

    def _split_df(self, chunk_size: int = 256, idcol: str = 'dataset_id'):
        """
        Разбивает DataFrame на чанки фиксированного размера.
        chunk_size - size rows in one chunk
        idcol - id col in dataset and adj_matrix
        """
        if (self.splited_data is not None) and (self.splited_data.is_cached):
            self.splited_data.unpersist()
        self.kgroups = int(np.ceil((self.count_df // chunk_size)))
        data = self.dataframe.withColumn('batch_id', F.row_number().over(Window.orderBy(F.rand())))

        data2 = data.filter(F.col('batch_id') > ((self.kgroups) * chunk_size)).withColumn('batch_id', F.lit(self.kgroups+1))
        data = data.filter(F.col('batch_id') <= ((self.kgroups) * chunk_size)).withColumn('batch_id', F.ntile(self.kgroups).over(Window.orderBy('batch_id')))
        data = data.union(data2)

        self.splited_data = data.repartitionByRange(self.kgroups+1, 'batch_id')
        self.splited_data.persist(StorageLevel.DISK_ONLY).foreach(lambda x: x)
        return None

    def get_df_generator(self, chunk_size:int = 256, idcol: str = 'dataset_id', num_epoch:int = 4, quadratic:bool = False):
        """
        Генерирует батчи данных с поддержкой эпох.
        chunk_size - size rows in one chunk
        idcol - id col in dataset and adj_matrix
        num_epoch - set num epoch generator
        """
        for i in range(num_epoch):
            self._split_df(chunk_size=chunk_size, idcol=idcol)
            self.items = self.splited_data.groupBy('batch_id').agg(F.collect_set(idcol).alias(idcol)).checkpoint()
            for chunk in range(1, self.kgroups+2):
                items = self.items.filter(F.col('batch_id') == chunk).select(idcol).collect()[0][0]
                if self.adjdataframe is not None:
                    if quadratic:
                        yield (i+1 , 
                               self.splited_data.filter(F.col('batch_id') == chunk).drop('batch_id').coalesce(1),
                               self.adjdataframe.filter(F.col(idcol).isin(items) & F.col(f'{idcol}_B').isin(items)).coalesce(1))
                    else:
                        yield (i+1 , 
                               self.splited_data.filter(F.col('batch_id') == chunk).drop('batch_id').coalesce(1),
                               self.adjdataframe.filter(F.col(idcol).isin(items)).coalesce(1))
                else:
                    yield (i+1 , self.splited_data.filter(F.col('batch_id') == chunk).drop('batch_id').coalesce(1))

class BatchedStreamProcessor:
    """Обработчик батчей с буферизацией и многопоточной загрузкой."""
    def __init__(self, spark: SparkSession, buffer_size=5):
        self.buffer_size = buffer_size
        self.spark = spark
        
    def load_batch(self, batch):
        if isinstance(batch, tuple):
            if len(batch) == 3:
                batch[1].persist(StorageLevel.MEMORY_ONLY).foreach(lambda x: x)
                batch[2].persist(StorageLevel.MEMORY_ONLY).foreach(lambda x: x)
                result = (batch[0], batch[1], batch[2])
            elif len(batch) == 2:
                batch[1].persist(StorageLevel.MEMORY_ONLY).foreach(lambda x: x)
                result = (batch[0], batch[1])
        else:
            batch.persist(StorageLevel.MEMORY_ONLY).foreach(lambda x: x)
            result = batch
        return result
    
    def output_batch(self, batch):
        if isinstance(batch, tuple):
            if len(batch) == 3:
                result = (batch[0], batch[1].toPandas(), batch[2].toPandas())
                batch[1].unpersist()
                batch[2].unpersist()
            elif len(batch) == 2:
                result = (batch[0], batch[1].toPandas())
                batch[1].unpersist()
        else:
            result = batch.toPandas()
            batch.unpersist()
        return result
    
    def batched_stream(self, batch_generator):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.buffer_size) as executor:
            batch_iterator = iter(batch_generator)
            futures = {}
            for batch in itertools.islice(batch_iterator, self.buffer_size):
                future = executor.submit(self.load_batch, batch)
                futures[future] = batch
            while futures:
                done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    batch = futures.pop(future)
                    try:
                        yield self.output_batch(future.result())
                    except Exception as e:
                        print(f'Error in batch: {e}')
                    try:
                        next_batch = next(batch_iterator)
                        new_future = executor.submit(self.load_batch, next_batch)
                        futures[new_future] = next_batch
                    except StopIteration:
                        pass

class SparkSimilarMatrix:
    """
    Вычисление матрицы схожести в распределенном режиме.
    
    Пример использования:
    >>> sim = SparkSimilarMatrix(spark, train_df)
    >>> sim.prep()
    >>> sim.create_sim_matrix(metric='cosine')
    >>> sim.create_degree_matrix(treshold=0.75)
    """
    def __init__(self,
                 spark: SparkSession,
                 train_df: DataFrame) -> None:
        self.spark = spark
        self.train_df = train_df
        self.sim_matrix = None
        self._idcol = None
        self._feature_columns = self.train_df.columns
        
    def prep(self, window_size = None, idcol: str = 'id') -> None:
        """Подготовка данных к расчету матрицы."""
        self._idcol = idcol
        
        if self._idcol not in self.train_df.columns:
            self.train_df = self.train_df.withColumn(self._idcol, F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))
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

        self._train_chunks = self.train_df.select('rank').distinct().rdd.flatMap(lambda x: x).collect()
        self.chunk_combinations = list(itertools.product(self._train_chunks, repeat=2))
        self.chunk_combinations = [i for i in self.chunk_combinations if i[0] <= i[1]]
        return None
        
    def create_sim_matrix(self, metric:str = 'euclidean') -> None:
        """Расчет матрицы схожести с выбранной метрикой."""
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

        def process_chunk_pair(pair, metric):
            chunk_a, chunk_b = pair
            df_a = self.train_df.filter(F.col('rank') == chunk_a)#.hint('broadcast')
            df_b = self.train_df.filter(F.col('rank') == chunk_b)#.hint('broadcast')
            #######################
            #df_a = F.broadcast(df_a)
            #df_b = F.broadcast(df_b)
            #######################
            
            #.withColumn('dst_weight', F.sqrt(F.col(f'a.dist_weight') * F.col(f'b.dist_weight')))\
            
            result = df_a.alias('a').crossJoin(df_b.alias('b'))\
                        .filter(F.col(f'a.{self._idcol}') < F.col(f'b.{self._idcol}'))\
                        .select(
                            *[F.col(f'a.{self._idcol}').alias(f'{self._idcol}_A'),
                            F.col(f'b.{self._idcol}').alias(f'{self._idcol}_B'),
                            F.col(f'a.features').alias(f'featuresA'),
                            F.col(f'b.features').alias(f'featuresB')]
                        )
            result = result.mapInPandas(calculate_distances, 
                                      result.withColumn('distance', F.lit(90.912591951925951)).schema)
            result = result.select(F.col(f'{self._idcol}_A'),F.col(f'{self._idcol}_B'),'distance').coalesce(1)
            result.persist(StorageLevel.DISK_ONLY).foreach(lambda x: x)
            return result
        
        results = []
        errors = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
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
            with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
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

    def create_degree_matrix(self, treshold:float = 0.3) -> None:
        """Создание матрицы степеней узлов."""
        if self.sim_matrix == None:
            raise ValueError('sim_matrix must be calculated first')
        
        sim_matrix_filtred = self.sim_matrix.filter(F.col('distance') >= treshold)
        self.degree_matrix = sim_matrix_filtred.select(
            F.col(f'{self._idcol}_A').alias('node'), F.col('distance'))\
            .union(sim_matrix_filtred.filter(F.col(f'{self._idcol}_A') != F.col(f'{self._idcol}_B'))\
                   .select(F.col(f'{self._idcol}_B').alias('node'), F.col('distance')))\
            .groupBy('node')\
            .agg(F.sum('distance').alias('degree'))\
            .orderBy('node')\
            .withColumn('degree', 1/F.sqrt('degree'))\
            .checkpoint()
        partnum = self.degree_matrix.select('node').distinct().count()
        
        self.sim_matrix_filtred = sim_matrix_filtred\
            .join(self.degree_matrix.select(
                F.col('node').alias(f'{self._idcol}_A'), 
                F.col('degree').alias('degree_A')), 
                on=f'{self._idcol}_A', how='left')\
            .join(self.degree_matrix.select(
                F.col('node').alias(f'{self._idcol}_B'), 
                F.col('degree').alias('degree_B')), 
                on=f'{self._idcol}_B', how='left')\
            .withColumn('distance', F.col('distance') * F.col('degree_A') * F.col('degree_B'))\
            .select(f'{self._idcol}_A', f'{self._idcol}_B', 'distance')\
            .repartitionByRange(partnum, f'{self._idcol}_A').checkpoint()