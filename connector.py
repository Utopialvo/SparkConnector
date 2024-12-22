import pandas as pd
import numpy as np

import datetime
import re
import os
import glob
import sys
import math

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.storagelevel import StorageLevel
from pyspark.serializers import MarshalSerializer
import pyspark.sql.functions as F
import pyspark.sql.types as T


class spark_connector:
    """
    Класс стандартного подключения к Spark

    exec_inst - количество контейнеров
    exec_cores - количество ядер в контейнере
    aloc_max - максимальное количество ядер для dynamic_allocation (если он вкл)
    memory_exec - количество памяти в контейнере
    name - название подключения для упрощения логгирования подключений в кластере
    rootdir - корневая папка в HDFS 
    enable_graphframes - включить работу с graphframes
    enable_clickhouse - включить работу с clickhouse
    enable_postgres - включить работу с postgres
    dynamic_allocation - включить динамическое выделение ресурсов
    intensive_mode - включить мод с быстрой очисткой памяти (нужно тюнить параметры для конкретных случаев)
    marshal_mode - подключить Marshal Serializer вместо Kryo Serializer


    Для подключения (standalone, yarn, k8s) нужно модернизировать самостоятельно в зависимости от того, как был развернут кластер.
    To Do:
    spark.master=yarn
    spark.deploy-mode=client

    SPARK_HOME
    JAVA_HOME
    PYSPARK_DRIVER_PYTHON
    PYSPARK_PYTHON
    HADOOP_HOME=/usr/local/hadoop-2.7.3
    HADOOP_CONF_HOME=/usr/local/hadoop-2.7.3/etc/hadoop
    HADOOP_CONF_DIR=/usr/local/hadoop-2.7.3/etc/hadoop
    HADOOP_PREFIX=/usr/local/hadoop-2.7.3
    .remote("sc://<sc_host>:<sc_port>")

    fair_scheduler.xml - можно меять под себя.
    """
    def __init__(self,
                 exec_inst: int = 100,
                 exec_cores: int = 5,
                 aloc_max: int = 500,
                 memory_exec: int = 25,
                 name: object = 'utopialvo',
                 rootdir: object = 'remote_dir_in_hdft',
                 enable_graphframes: bool = False,
                 enable_clickhouse: bool = False,
                 enable_postgres: bool = False,
                 dynamic_allocation: bool = False,
                 intensive_mode: bool = False,
                 marshal_mode: bool = True) -> None:
        
        os.environ['SPARK_MAJOR_VERSION'] = '3'
        os.environ['SPARK_HOME'] = '/usr/local/spark'
        os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-1.17.0-openjdk-amd64'
        os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
        os.environ['PYSPARK_PYTHON'] = sys.executable
        os.environ['PYARROW_IGNORE_TIMEZONE'] = '1'

        if dynamic_allocation:
            aloc_max = int(aloc_max / exec_cores)
            spark_param_shuffle = int(aloc_max * exec_cores * 2.6)
        else:
            aloc_max = exec_inst
            spark_param_shuffle = int(exec_inst * exec_cores * 2.6)
        chkp = 'CheckpointDir/' + name
        name = name + '_' + str(datetime.datetime.now()).replace(' ', '_')
        overhead = f'{int((memory_exec * 1024) * 0.10)}m'
        conf = SparkConf()\
                .setAppName(f'{name}')\
                .setMaster('yarn')\
                .set('spark.log.level', 'ERROR')\
                .set('spark.executor.instances', exec_inst)\
                .set('spark.executor.cores', exec_cores)\
                .set('spark.default.parallelism', spark_param_shuffle)\
                .set('spark.sql.shuffle.partitions', spark_param_shuffle)\
                .set('spark.executor.memory', f'{memory_exec}g')\
                .set('spark.executor.memoryOverhead', overhead)\
                .set('spark.driver.cores', 2)\
                .set('spark.driver.memory', f'{memory_exec}g')\
                .set('spark.driver.maxResultSize', f'{memory_exec}g')\
                .set('spark.scheduler.mode', f'FAIR')\
                .set('spark.scheduler.allocation.file', f'file://{glob.glob(f"{os.getcwd()}/fair*.xml")[0]}')\
                .set('spark.sql.broadcastTimeout', '36000')\
                .set('spark.hadoop.mapreduce.input.fileinputformat.input.dir.recursive', 'true')\
                .set('spark.sql.catalogImplementation', 'hive')\
                .set('spark.sql.hive.convertMetastoreParquet', 'false')\
                .set('spark.sql.hive.metastorePartitionPruning', 'true')\
                .set('spark.sql.parquet.writeLegacyFormat', 'true')\
                .set('spark.sql.parquet.binaryAsString', 'true')\
                .set('spark.sql.sources.partitionOverwriteMode', 'dynamic')\
                .set('spark.locality.wait', '2')\
                .set('spark.port.maxRetries', '150')\
                .set('spark.sql.adaptive.enabled', 'true')\
                .set('spark.sql.autoBroadcastJoinThreshold', '104857600')\
                .set('spark.sql.adaptive.coalescePartitions.enabled', 'true')\
                .set('spark.sql.adaptive.coalescePartitions.parallelismFirst', 'true')\
                .set('spark.sql.adaptive.coalescePartitions.initialPartitionNum', spark_param_shuffle * 100)\
                .set('spark.sql.adaptive.coalescePartitions.minPartitionSize', '8MB')\
                .set('spark.sql.adaptive.skewJoin.enabled', 'true')\
                .set('spark.sql.adaptive.forceOptimizeSkewedJoin', 'true')\
                .set('spark.sql.adaptive.optimizeSkewsInRebalancePartitions.enabled', 'true')\
                .set('spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes', '144MB')\
                .set('spark.sql.adaptive.skewJoin.skewedPartitionFactor', '2.0')\
                .set('spark.sql.adaptive.localShuffleReader.enabled', 'true')\
                .set('spark.sql.adaptive.advisoryPartitionSizeInBytes', '64MB')\
                .set('spark.sql.parquet.compression.codec', 'snappy')\
                .set('spark.sql.execution.arrow.pyspark.enabled', 'true')\
                .set('spark.sql.execution.arrow.pyspark.fallback.enabled', 'true')\
                .set('spark.rdd.compress', 'true')\
                .set('spark.shuffle.compress', 'true')\
                .set('spark.shuffle.spill.compress', 'true')\
                .set('spark.shuffle.file.buffer', '1600k')\
                .set('spark.shuffle.unsafe.file.output.buffer', '1600k')\
                .set('spark.reducer.maxSizeInFlight', '128m')\
                .set('spark.checkpoint.compress', 'true')\
                .set('spark.sql.files.maxPartitionBytes', '134217728')\
                .set('spark.sql.files.openCostInBytes', '134217728')\
                .set('spark.memory.fraction', '0.75')\
                .set('spark.memory.storageFraction', '0.20')\
                .set('spark.memory.offHeap.use', 'False')\
                .set('spark.rpc.message.maxSize', '2047')
        
        if dynamic_allocation:
            conf = conf\
                .set('spark.dynamicAllocation.enabled', 'true')\
                .set('spark.shuffle.service.enabled', 'true')\
                .set('spark.dynamicAllocation.maxExecutors', f'{aloc_max}')\
                .set('spark.dynamicAllocation.minExecutors', '1')\
                .set('spark.dynamicAllocation.executorIdleTimeout', '15s')

        
        if enable_graphframes or enable_clickhouse or enable_postgres:
            jar_files = []
            if enable_graphframes:
                jar_files.append(glob.glob(f'{os.getcwd()}/graphf*.jar')[0])
            if enable_clickhouse:
                jar_files.append(glob.glob(f'{os.getcwd()}/clickhouse-jdbc*.jar')[0])
            if enable_postgres:
                jar_files.append(glob.glob(f'{os.getcwd()}/postgres*.jar')[0])
            conf = conf.set("spark.jars", ",".join(jar_files))

        if intensive_mode:
            conf = conf\
                .set('spark.memory.fraction', '0.25')\
                .set('spark.memory.storageFraction', '0.3')\
                .set('spark.executor.extraJavaOptions', '-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35 -XX:ConcGCThreads=12')\
                .set('spark.cleaner.periodicGC.interval', '15min')
        
        if marshal_mode:
            self._sc = SparkContext(conf= conf, serializer=MarshalSerializer())
        else:
            conf = conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            self._sc = SparkContext(conf=conf)
        self._sc.setLogLevel('ERROR')
        self._sc.setCheckpointDir(chkp)
        self.spark = SparkSession(self._sc)
        self.partition_param = spark_param_shuffle
        self.home_dir = rootdir

        if enable_graphframes:
            self.spark.sparkContext.addPyFile(glob.glob(f'{os.getcwd()}/graphf*.zip')[0])


    
    def _estimate_size_in_bytes(self, df: DataFrame):
        """
        Метод для оценки занимаемой датафреймом памяти 
        """
        return self.spark._jsparkSession.sessionState()\
                    .executePlan(df._jdf.queryExecution().logical(), df._jdf.queryExecution().mode())\
                    .optimizedPlan()\
                    .stats()\
                    .sizeInBytes()


    def repartitioning(self,
                           df: DataFrame,
                           mode: str = 'parquet',
                           partititon_cols: list = None,
                           block_size: int = (128 * 1024 * 1024),
                           serialization_factor: int = 8) -> DataFrame:
        """
        Метод для партиционирования датафрейма по заданному размеру
        mode - может быть ['memory', 'parquet']. Запись на диск или кэширование в памяти.
        block_size - размер блока в HDFS
        serialization_factor - коэффициент, для оптимальной записи в parquet
        """
        assert mode in ['memory', 'parquet']
        if serialization_factor < 1:
            raise ValueError('serialization_factor must be greater or equal to 1')

        caching_required = not df.is_cached
        
        if caching_required:
            df.persist().foreach(lambda x: x)
            
        df_size_in_bytes = self._estimate_size_in_bytes(df)

        if mode == 'parquet':
            no_part = math.ceil(df_size_in_bytes / block_size / serialization_factor)
        else:
            no_part = math.ceil(df_size_in_bytes / block_size)

        if caching_required:
            df.unpersist()

        if no_part == 1:
            result = df.coalesce(1)
        else:
            if type(partititon_cols) == type(None):
                result = df.repartition(no_part)
            else:
                result = df.repartition(no_part, partititon_cols)
        return result


    def repartition_and_checkpoint(self,
                                  df: DataFrame,
                                  partititon_cols: list = None,
                                  cache: bool = True) -> DataFrame:
        """
        Метод для создания чекпоинта, оптимального по размеру партиций для дальнейшей работы.
        partititon_cols - название колонки партиционирования или лист колонок.
        cache - запись через оперативную память или через запись на диск.
        """
        if cache:
            df.persist().foreach(lambda x: x)
            df_ = self.repartitioning(df, mode = 'memory', partititon_cols = partititon_cols)
        else:
            df.persist(StorageLevel.DISK_ONLY).foreach(lambda x: x)
            df_ = self.repartitioning(df, mode = 'parquet', partititon_cols = partititon_cols)
        
        df_ = df_.checkpoint(False)
        df_.foreach(lambda x: x)
        df.unpersist()
        return df_
    
    def read_table(self,
                   table:str,
                   DB: str = None) -> DataFrame:
        """
        Чтение из HDFS
        DB - путь к корневой папке в HDFS. Если задан при инициализации объекта, то можно указывать только путь к таблице.
        table - путь к таблице относительно корневой папки в HDFS.
        """
        DB = self.home_dir if DB is None else DB
        self.spark.catalog.refreshTable(f'{DB}.{table}')
        return self.spark.read.table((f'{DB}.{table}'))

    def clickhouse_read_table(self,
                   url:str,
                   table:str,
                   user:str,
                   password:str,
                   driver:str = "com.clickhouse.jdbc.ClickHouseDriver") -> DataFrame:
        """
        Чтение из clickhouse
        url = "jdbc:clickhouse://<host>:<port>/<database>"
        table = "<table_name>"
        user = "<username>"
        password = "<password>"
        driver = "com.clickhouse.jdbc.ClickHouseDriver"
        """
        return self.spark.read.format("jdbc") \
                    .option('driver', driver) \
                    .option("url", url) \
                    .option("dbtable", table) \
                    .option("user", user) \
                    .option("password", password) \
                    .load()
        
    def postgres_read_table(self,
                   url:str,
                   table:str,
                   user:str,
                   password:str,
                   driver:str = "org.postgresql.Driver") -> DataFrame:
        """
        Чтение из postgres
        url = путь/ДБ "jdbc:postgresql://<host>:<port>/<database>"
        table = название таблицы
        user = "postgres"
        password = "password"
        driver = "org.postgresql.Driver"
        """
        properties = {"user": user,"password": password,"driver": driver}
        return self.spark.read.jdbc(url, table, properties=properties)

    def save_table(self,
                   df: DataFrame,
                   table: str,
                   save_mode: str = 'overwrite',
                   partitionBy: list = None):
        """
        Запись в HDFS
        table - полный путь к таблице
        save_mode - перезапись таблицы или добавление строк в существующую ['overwrite', 'append']
        partitionBy - ключ партиционирования
        """
        assert save_mode in ['overwrite', 'append']

        if save_mode == 'overwrite':
            if (isinstance(partitionBy, list)):
                for i in partitionBy:
                    if not partitionBy in df.columns:
                        raise AttributeError(f'Invalid column: {i}')
            elif (isinstance(partitionBy, str)):
                if not partitionBy in df.columns:
                    raise AttributeError(f'Invalid column: {partitionBy}')
            elif type(partitionBy) == type(None):
                pass
            else:
                raise AttributeError(f'partitionBy not list or str')
            df.write.partitionBy(partitionBy).mode(save_mode).format('parquet').saveAsTable(table)
        else:
            df.write.partitionBy(partitionBy).mode(save_mode).insertInto(table)

    def clickhouse_save_table(self,
                   df: DataFrame,
                   table: str,
                   url: str,
                   orderBy: str,
                   user: str,
                   password: str,
                   save_mode: str = 'overwrite',
                   driver:str = "com.clickhouse.jdbc.ClickHouseDriver"):
        """
        Запись в ClickHouse
        url = "jdbc:clickhouse://<host>:<port>/<database>"
        table = "<table_name>"
        orderBy = clickhouse MergeTree must be set order by (ID), where ID is orderBy arg
        user = "<username>"
        password = "<password>"
        driver = "com.clickhouse.jdbc.ClickHouseDriver"
        """
        df.write.format("jdbc") \
            .option("driver", driver) \
            .option("url", url) \
            .option("dbtable", table) \
            .option("createTableOptions", f"engine=MergeTree() order by ({orderBy})")\
            .option("user", user) \
            .option("password", password) \
            .mode(save_mode) \
            .save()

    def postgres_save_table(self,
               df: DataFrame,
               table: str,
               url: str,
               user: str,
               password: str,
               save_mode: str = 'overwrite',
               driver:str = "org.postgresql.Driver"):
        """
        Запись в Postgres
        url = путь/ДБ "jdbc:postgresql://<host>:<port>/<database>"
        table = название таблицы
        user = "postgres"
        password = "password"
        driver = "org.postgresql.Driver"
        """
        properties = {"user": user,"password": password,"driver": driver}
        df.write.jdbc(url=url, table=table, mode=save_mode, properties=properties)

    def del_Checkpoints(self) -> None:
        """
        Метод очищает папку с чекпоинтами
        """
        try:
            fs = self._sc._jvm.org.apache.hadoop.fs.FileSystem.get(self._sc._jsc.hadoopConfiguration())
            cd = self._sc._jvm.org.apache.hadoop.fs.Path(str(self._sc._jsc.sc().getCheckpointDir().get()))
            if fs.exists(cd):
                fs.delete(cd)
                print('del')
        except:
            print('dont del')
            
    def stop_spark(self) -> None:
        """
        Метод закрывает сессию спарка и очищает папку с чекпоинтами
        """
        print(f'time: {str(datetime.datetime.now())}')
        self.del_Checkpoints()
        try:
            self.spark.stop()
            return print('stop')
        except:
            return print(f'dont stop me now')
            
    def __del__(self) -> None:
        self.stop_spark()
