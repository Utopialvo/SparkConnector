from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from typing import Union, Dict, List, Optional
import joblib
import pandas as pd

class Normalizer:
    def __init__(
        self,
        method: Optional[str] = None,
        columns: List[str] = None,
        methods: Dict[str, str] = None
    ):
        self.valid_methods = ['minmax', 'zscore', 'range']
        
        if methods is None and (method is None or columns is None):
            self.methods = {}
            self.columns = []
            self.stats = {}
            return None
        
        if methods is not None:
            if not isinstance(methods, dict):
                raise ValueError("Methods must be a dictionary")
            for m in methods.values():
                if m not in self.valid_methods:
                    raise ValueError(f"Invalid method: {m}")
            self.methods = methods
            self.columns = list(methods.keys())
        else:
            if method is None or columns is None:
                raise ValueError("Either provide 'methods' or both 'method' and 'columns'")
            if method not in self.valid_methods:
                raise ValueError(f"Invalid method: {method}")
            self.methods = {col: method for col in columns}
            self.columns = columns
        
        self.stats = {}

    def fit(self, train_df: DataFrame) -> 'Normalizer':
        """Собирает статистику о датасете согласно конфигурации."""
        missing_cols = [col for col in self.columns if col not in train_df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")

        agg_exprs = []
        for col, method in self.methods.items():
            if method == 'minmax':
                agg_exprs.extend([
                    F.min(col).alias(f'min_{col}'),
                    F.max(col).alias(f'max_{col}')
                ])
            elif method == 'zscore':
                agg_exprs.extend([
                    F.avg(col).alias(f'mean_{col}'),
                    F.stddev(col).alias(f'stddev_{col}')
                ])
            elif method == 'range':
                agg_exprs.extend([
                    F.min(col).alias(f'min_{col}'),
                    F.max(col).alias(f'max_{col}'),
                    F.avg(col).alias(f'mean_{col}')
                ])

        stats_row = train_df.select(agg_exprs).first()

        for col, method in self.methods.items():
            if method == 'minmax':
                self.stats[col] = {
                    'method': method,
                    'params': (
                        stats_row[f'min_{col}'],
                        stats_row[f'max_{col}']
                    )
                }
            elif method == 'zscore':
                self.stats[col] = {
                    'method': method,
                    'params': (
                        stats_row[f'mean_{col}'],
                        stats_row[f'stddev_{col}']
                    )
                }
            elif method == 'range':
                self.stats[col] = {
                    'method': method,
                    'params': (
                        stats_row[f'min_{col}'],
                        stats_row[f'max_{col}'],
                        stats_row[f'mean_{col}']
                    )
                }
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        """Преобразует датафрейм согласно собранной статистике."""
        if not self.stats:
            raise RuntimeError("Fit the normalizer first")
        
        df_transformed = df
        for col in self.columns:
            stat = self.stats.get(col)
            if not stat:
                raise ValueError(f"No stats for column: {col}")
            
            method = stat['method']
            params = stat['params']
            
            if method == 'minmax':
                min_val, max_val = params
                if max_val == min_val:
                    df_transformed = df_transformed.withColumn(col, F.lit(0.0))
                else:
                    df_transformed = df_transformed.withColumn(
                        col, 
                        (F.col(col) - min_val) / (max_val - min_val))
            elif method == 'zscore':
                mean_val, stddev_val = params
                if stddev_val == 0:
                    stddev_val = 1e-19
                df_transformed = df_transformed.withColumn(
                    col, 
                    ((F.col(col) - mean_val) / stddev_val))
            elif method == 'range':
                min_val, max_val, mean_val = params
                range_val = max_val - min_val
                if range_val == 0:
                    df_transformed = df_transformed.withColumn(col, F.lit(0.0))
                else:
                    df_transformed = df_transformed.withColumn(
                        col, 
                        ((F.col(col) - mean_val) / range_val))
        return df_transformed

    def save(self, path: str) -> None:
        """Сохраняет параметры нормализации в файл"""
        data = {
            'methods': self.methods,
            'stats': self.stats,
            'columns': self.columns,
        }
        joblib.dump(data, path)

    def load(self, path: str) -> None:
        """Загружает параметры из файла и возвращает инициализированный объект"""
        data = joblib.load(path)
        self.methods = data['methods']
        self.stats = data['stats']
        self.columns = data['columns']
        


class NormalizerPandas:
    def __init__(
        self,
        method: Optional[str] = None,
        columns: Optional[List[str]] = None,
        methods: Optional[Dict[str, str]] = None
    ):
        self.valid_methods = ['minmax', 'zscore', 'range']
        
        if methods is None and (method is None or columns is None):
            self.methods = {}
            self.columns = []
            self.stats = {}
            return None
        
        if methods is not None:
            if not isinstance(methods, dict):
                raise ValueError("Methods must be a dictionary")
            for m in methods.values():
                if m not in self.valid_methods:
                    raise ValueError(f"Invalid method: {m}")
            self.methods = methods
            self.columns = list(methods.keys())
        else:
            if method is None or columns is None:
                raise ValueError("Either provide 'methods' or both 'method' and 'columns'")
            if method not in self.valid_methods:
                raise ValueError(f"Invalid method: {method}")
            self.methods = {col: method for col in columns}
            self.columns = columns
        
        self.stats = {}

    def fit(self, df: pd.DataFrame) -> 'NormalizerPandas':
        """Собирает статистику о датасете согласно конфигурации."""
        missing_cols = [col for col in self.columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")

        for col, method in self.methods.items():
            if method == 'minmax':
                self.stats[col] = {
                    'method': method,
                    'min': df[col].min(),
                    'max': df[col].max()
                }
            elif method == 'zscore':
                self.stats[col] = {
                    'method': method,
                    'mean': df[col].mean(),
                    'std': df[col].std()
                }
            elif method == 'range':
                self.stats[col] = {
                    'method': method,
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean()
                }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Преобразует датафрейм согласно собранной статистике."""
        if not self.stats:
            raise RuntimeError("Fit the normalizer first")

        df = df.copy()
        for col in self.columns:
            stat = self.stats.get(col)
            if not stat:
                raise ValueError(f"No stats for column: {col}")

            method = stat['method']
            
            if method == 'minmax':
                min_val = stat['min']
                max_val = stat['max']
                if max_val == min_val:
                    df[col] = 0.0
                else:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
            
            elif method == 'zscore':
                mean_val = stat['mean']
                std_val = stat['std']
                if std_val == 0:
                    std_val = 1e-19
                df[col] = (df[col] - mean_val) / std_val
            
            elif method == 'range':
                min_val = stat['min']
                max_val = stat['max']
                mean_val = stat['mean']
                range_val = max_val - min_val
                if range_val == 0:
                    df[col] = 0.0
                else:
                    df[col] = (df[col] - mean_val) / range_val
        return df

    def save(self, path: str) -> None:
        """Сохраняет параметры нормализации в файл"""
        data = {
            'methods': self.methods,
            'stats': self.stats,
            'columns': self.columns,
        }
        joblib.dump(data, path)

    def load(self, path: str) -> None:
        """Загружает параметры из файла и возвращает инициализированный объект"""
        data = joblib.load(path)
        self.methods = data['methods']
        self.stats = data['stats']
        self.columns = data['columns']