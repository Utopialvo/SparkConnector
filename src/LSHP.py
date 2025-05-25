from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T

class LSHSampler:
    """Locality Sensitive Hashing (LSH) with density sampling
    
    Args:
        input_cols: List of feature column names
        num_hashes: Number of hash functions to use
        bin_widths: List of bucket width values
        random_proj: Use random normal projections (True) or Cauchy (False)
        seed: Random seed for reproducibility
        output_prefix: Prefix for output bucket columns
    """
    
    def __init__(
        self,
        input_cols: List[str],
        num_hashes: int = 35,
        bin_widths: Optional[List[float]] = None,
        random_proj: bool = False,
        seed: Optional[int] = None,
        output_prefix: str = "bins"
    ):
        if num_hashes <= 0:
            raise ValueError("num_hashes must be positive")
        if not bin_widths:
            raise ValueError("bin_widths cannot be empty")
            
        self.dim = len(input_cols)
        self.num_hashes = num_hashes
        self.bin_widths = bin_widths
        self.random_proj = random_proj
        self.input_cols = input_cols
        self.output_prefix = output_prefix
        self.seed = seed
        self.projections = self._generate_projections()
        self._projections_bcast = None

    def _generate_projections(self) -> List[List[float]]:
        """Generate projection vectors for LSH bucketing
        """
        if self.seed is not None:
            np.random.seed(self.seed)
            
        # Generate random or Cauchy distributed projections
        if self.random_proj:
            projections = np.random.randn(self.num_hashes, self.dim)
            norms = np.linalg.norm(projections, axis=1, keepdims=True)
            norms[norms == 0] = 1e-8  # Prevent division by zero
            projections /= norms
        else:
            projections = np.random.standard_cauchy(size=(self.num_hashes, self.dim))
            mad = np.median(np.abs(projections), axis=1)  # Median absolute deviation
            mad[mad == 0] = 1e-8
            projections /= mad[:, np.newaxis]
            
        return projections.astype(np.float32).tolist()

    def _prepare_broadcast(self, spark: SparkSession) -> None:
        """Broadcast projection vectors"""
        if self._projections_bcast is None:
            self._projections_bcast = spark.sparkContext.broadcast(self.projections)

    @property
    def _output_schema(self) -> T.StructType:
        """Schema definition for LSH bucket columns"""
        return T.StructType([
            T.StructField(f"{self.output_prefix}_{i}", T.ArrayType(T.IntegerType()))
            for i in range(len(self.bin_widths))
        ])

    def _create_udf(self) -> F.UserDefinedFunction:
        """Create Pandas UDF"""
        proj = self._projections_bcast
        bin_widths = self.bin_widths
        prefix = self.output_prefix

        @F.pandas_udf(self._output_schema, F.PandasUDFType.SCALAR)
        def _lsh_udf(*cols: pd.Series) -> pd.DataFrame:
            projections = np.array(proj.value, dtype=np.float32)
            data = np.column_stack(cols).astype(np.float32)
            dots = data @ projections.T
            return pd.DataFrame({
                f"{prefix}_{i}": (dots // width).astype(int).tolist()
                for i, width in enumerate(bin_widths)
            })
        return _lsh_udf

    def transform(self, df: DataFrame) -> DataFrame:
        """Apply LSH transformation to input DataFrame
        Returns:
            Original DataFrame with additional bucket columns
        """
        self._prepare_broadcast(df.sql_ctx.sparkSession)
        udf = self._create_udf()
        temp_col = f"__temp_{self.output_prefix}"
        output_cols = [
            F.col(temp_col).getItem(f"{self.output_prefix}_{i}").alias(f"{self.output_prefix}_{i}")
            for i in range(len(self.bin_widths))
        ]
        return df.withColumn(temp_col, udf(*self.input_cols)).select("*", *output_cols).drop(temp_col)

    def compute_bucket_sizes(self, df: DataFrame, bin_width_idx: int = 0) -> DataFrame:
        """Calculate bucket sizes for specified bin width
        
        Args:
            bin_width_idx: Index of bin width to use from bin_widths list
        """
        if bin_width_idx >= len(self.bin_widths):
            raise ValueError("Invalid bin_width_idx")
            
        bucket_col = f"{self.output_prefix}_{bin_width_idx}"
        return (
            self.transform(df)
            .withColumn("bucket_key", F.concat_ws("|", F.col(bucket_col)))
            .groupBy("bucket_key")
            .agg(F.count("*").alias("bucket_size"))
        )

    def add_density_weights(
        self, 
        df: DataFrame, 
        inverse_density: bool = True, 
        bin_width_idx: int = 0,
        eps: float = 1e-10
    ) -> DataFrame:
        """Add sampling weights based on bucket densities
        Args:
            inverse_density: Use inverse density weighting (1/bucket_size)
            bin_width_idx: Index of bin width to use for density calculation
            eps: Small value to prevent division by zero
        """
        bucket_sizes = self.compute_bucket_sizes(df, bin_width_idx)
        bucket_col = f"{self.output_prefix}_{bin_width_idx}"
        
        df_with_buckets = self.transform(df).withColumn("bucket_key", F.concat_ws("|", F.col(bucket_col)))
        df_with_density = df_with_buckets.join(F.broadcast(bucket_sizes), "bucket_key", "left")
        
        weight_expr = (1.0 / (F.col("bucket_size") + eps)) if inverse_density else F.col("bucket_size")
        return df_with_density.withColumn("weight", weight_expr)

    def sample(
        self, 
        df: DataFrame, 
        n_samples: int, 
        inverse_density: bool = True, 
        bin_width_idx: int = 0
    ) -> DataFrame:
        """Density sampling
        
        Args:
            n_samples: Target number of samples to return
            inverse_density: Sample inversely proportional to density
            bin_width_idx: Index of bin width to use for sampling
        """
        df_with_weights = self.add_density_weights(df, inverse_density, bin_width_idx)
        
        bucket_stats = df_with_weights.groupBy("bucket_key").agg(
            F.sum("weight").alias("sum_weight"),
            F.count("*").alias("count")
        ).collect()
        
        total_weight = df_with_weights.select(F.sum("weight")).first()[0] or 0.0
        fractions = {}
        for row in bucket_stats:
            bucket_key = row["bucket_key"]
            sum_weight = row["sum_weight"] or 0.0
            count = row["count"] or 0
            if total_weight > 0 and count > 0:
                frac = (sum_weight / total_weight) * (n_samples / count)
                fractions[bucket_key] = min(frac, 1.0)
            else:
                fractions[bucket_key] = 0.0
                
        if not fractions:
            return df_with_weights.limit(0)
            
        return (
            df_with_weights
            .sampleBy("bucket_key", fractions, seed=self.seed)
            .drop("weight", "bucket_size", "bucket_key")
        )