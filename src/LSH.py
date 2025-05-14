from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T


class MultiBucketLSH:
    """Locality Sensitive Hashing (LSH) with multiple bucket widths for Spark DataFrames.
    
    This class implements multi-probe LSH using either random projections or Cauchy projections,
    allowing for multiple bucket widths to be used simultaneously.
    
    Args:
        input_cols: List of column names to use as input features.
        dim: Dimensionality of the input data. Default is 2.
        num_hashes: Number of hash functions to use. Default is 35.
        bin_widths: List of bucket widths to use. Default is [1.0, 5.0].
        random_proj: Whether to use random normal projections (True) or 
                    Cauchy projections (False). Default is False.
        seed: Random seed for reproducibility. Default is None.
        output_prefix: Prefix for output column names. Default is "bins".
    """
    
    def __init__(
        self,
        input_cols: List[str],
        dim: int = 2,
        num_hashes: int = 35,
        bin_widths: Optional[List[float]] = None,
        random_proj: bool = False,
        seed: Optional[int] = None,
        output_prefix: str = "bins"
    ):
        self.dim = dim
        self.num_hashes = num_hashes
        self.bin_widths = bin_widths or [1.0, 5.0]
        self.random_proj = random_proj
        self.input_cols = input_cols
        self.output_prefix = output_prefix
        self.seed = seed
        self.projections = self._generate_projections()
        self._projections_bcast = None

    def _prepare_broadcast(self, spark: SparkSession) -> None:
        """Prepare broadcast variable for projections if not already done."""
        if self._projections_bcast is None:
            self._projections_bcast = spark.sparkContext.broadcast(self.projections)

    def _generate_projections(self) -> List[List[float]]:
        """Generate projection vectors based on initialization parameters.
        
        Returns:
            List of projection vectors suitable for broadcasting.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
            
        if self.random_proj:
            projections = np.random.randn(self.num_hashes, self.dim)
            projections /= np.linalg.norm(projections, axis=1, keepdims=True)
        else:
            projections = np.random.standard_cauchy(size=(self.num_hashes, self.dim))
        
        return projections.astype(np.float32).tolist()

    @property
    def _output_schema(self) -> T.StructType:
        """Define the output schema for the UDF."""
        return T.StructType([
            T.StructField(f"{self.output_prefix}_{i}", T.ArrayType(T.IntegerType()))
            for i in range(len(self.bin_widths))
        ])
    
    def _create_udf(self) -> F.PandasUDFType:
        """Create and return a Pandas UDF for LSH bucketing."""
        proj = self._projections_bcast
        bin_widths = self.bin_widths
        prefix = self.output_prefix
        
        @F.pandas_udf(self._output_schema, F.PandasUDFType.SCALAR)
        def _lsh_udf(*cols: pd.Series) -> pd.DataFrame:
            """Compute LSH buckets for input columns.
            
            Args:
                cols: Input pandas Series representing columns from Spark DataFrame.
                
            Returns:
                DataFrame with bucket assignments for each specified bin width.
            """
            projections = np.array(proj.value, dtype=np.float32)
            data = np.column_stack(cols).astype(np.float32)
            
            # Vectorized computation of all projections
            dots = data @ projections.T
            
            # Compute buckets for all widths
            results = {
                f"{prefix}_{i}": (dots // width).astype(int).tolist()
                for i, width in enumerate(bin_widths)
            }
            
            return pd.DataFrame(results)
        
        return _lsh_udf
    
    def transform(self, df: DataFrame) -> DataFrame:
        """Apply LSH bucketing to the input DataFrame.
        
        Args:
            df: Input Spark DataFrame.
            
        Returns:
            DataFrame with additional columns containing bucket assignments.
        """
        self._prepare_broadcast(df.sql_ctx.sparkSession)
        udf = self._create_udf()
        temp_col = f"__temp_{self.output_prefix}"
        
        # Generate column expressions for the output
        output_cols = [
            F.col(temp_col).getItem(f"{self.output_prefix}_{i}").alias(f"{self.output_prefix}_{i}")
            for i in range(len(self.bin_widths))
        ]
        
        return (
            df
            .withColumn(temp_col, udf(*self.input_cols))
            .select("*", *output_cols)
            .drop(temp_col)
        )