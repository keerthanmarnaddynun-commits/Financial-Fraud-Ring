"""Centralised Apache Spark session factory with clean configuration and checkpointing."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import pyspark
from pyspark.sql import SparkSession

_spark_major = int(pyspark.__version__.split(".")[0])
_scala_version = "s_2.13" if _spark_major >= 4 else "s_2.12"
_default_gf_pkg = (
    f"graphframes:graphframes:0.8.3-spark3.5-{_scala_version}"
    if _spark_major >= 4
    else "io.graphframes:graphframes-spark3_2.12:0.9.3"
)
GRAPHFRAMES_PACKAGE = os.getenv("GRAPHFRAMES_PACKAGE", _default_gf_pkg)


def setup_hadoop_environment() -> None:
    """Propagate Python executable and configure HADOOP_HOME from environment only."""
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    hadoop_home = os.environ.get("HADOOP_HOME")
    if hadoop_home and os.path.exists(hadoop_home):
        bin_dir = os.path.join(hadoop_home, "bin")
        if os.path.exists(bin_dir) and bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{bin_dir};" + os.environ.get("PATH", "")


def create_spark_session(
    app_name: str = "PaySimFraudAnalytics",
    master: str = "local[*]",
    driver_memory: Optional[str] = None,
    shuffle_partitions: Optional[int | str] = None,
    local_dir: Optional[str] = None,
) -> SparkSession:
    """Start or retrieve a configured Spark session with checkpointing."""
    setup_hadoop_environment()

    mem = driver_memory or os.getenv("SPARK_DRIVER_MEMORY", "6g")
    partitions = str(shuffle_partitions or os.getenv("SPARK_SHUFFLE_PARTITIONS", "48"))
    spark_local_dir = local_dir or os.getenv("SPARK_LOCAL_DIRS")

    builder = (
        SparkSession.builder.master(master)
        .appName(app_name)
        .config("spark.jars.packages", GRAPHFRAMES_PACKAGE)
        .config("spark.driver.memory", mem)
        .config("spark.jars.excludes", "org.slf4j:slf4j-api,org.scala-lang:scala-library")
        .config("spark.default.parallelism", partitions)
        .config("spark.sql.shuffle.partitions", partitions)
        .config("spark.ui.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.adaptive.enabled", "true")
    )

    if spark_local_dir:
        builder = builder.config("spark.local.dir", spark_local_dir)

    spark = builder.getOrCreate()

    # Configure checkpoint directory for iterative graph algorithms
    checkpoint_dir = Path("runs") / "_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    spark.sparkContext.setCheckpointDir(str(checkpoint_dir.resolve()))

    return spark
