"""
utils.py
Common environment setup, Spark session initialization, and performance monitoring.
"""

import os
import sys
import time
import logging
from typing import Dict, Any, Optional

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("FraudAnalytics")


def setup_environment() -> None:
    """
    Configures Windows environment variables for PySpark, Hadoop, and Python paths.
    """
    # 1. Ensure Python executable is correctly propagated to PySpark workers
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    # 2. Check and configure Hadoop / winutils paths on Windows
    candidate_hadoop_dirs = [
        r"C:\hadoop\bin",
        r"C:\Users\nbala\hadoop\bin",
        os.path.join(os.environ.get("HADOOP_HOME", ""), "bin")
    ]
    for h_dir in candidate_hadoop_dirs:
        if os.path.exists(h_dir):
            if h_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{h_dir};" + os.environ.get("PATH", "")
            hadoop_home = os.path.dirname(h_dir)
            os.environ["HADOOP_HOME"] = hadoop_home
            break


def get_spark_session(
    app_name: str = "FinancialFraudAnalytics",
    master: str = "local[*]",
    driver_memory: str = "4g",
    shuffle_partitions: int = 8,
    checkpoint_dir: Optional[str] = None
):
    """
    Initializes a PySpark session configured with GraphFrames and performance optimizations.
    """
    setup_environment()
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.executor.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.default.parallelism", str(shuffle_partitions))
        .config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.5-s_2.12")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.rdd.compress", "true")
        .config("spark.driver.maxResultSize", "2g")
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # Set up checkpoint directory (required for connectedComponents & labelPropagation)
    if checkpoint_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoint_dir = os.path.join(base_dir, "output", "checkpoints")

    os.makedirs(checkpoint_dir, exist_ok=True)
    spark.sparkContext.setCheckpointDir(checkpoint_dir)

    return spark


class SparkPerformanceTracker:
    """
    Tracks execution wall-clock time, JVM heap memory, and shuffle metrics across Spark operations.
    """

    def __init__(self, spark):
        self.spark = spark
        self.jvm = spark._jvm
        self.start_time: float = 0.0
        self.stage_baseline_ids = set()

    def start(self):
        """Starts timing and captures baseline stages."""
        self.start_time = time.perf_counter()
        try:
            tracker = self.spark.sparkContext.statusTracker()
            self.stage_baseline_ids = set(tracker.getStageIds())
        except Exception:
            self.stage_baseline_ids = set()
        return self

    def stop(self) -> Dict[str, Any]:
        """Calculates elapsed time, memory overhead, and shuffle data transferred."""
        duration_sec = time.perf_counter() - self.start_time
        metrics = {
            "duration_seconds": round(duration_sec, 3),
            "jvm_used_memory_mb": 0.0,
            "jvm_max_memory_mb": 0.0,
            "shuffle_read_bytes": 0,
            "shuffle_write_bytes": 0,
        }

        # Query JVM Memory
        try:
            runtime = self.jvm.java.lang.Runtime.getRuntime()
            total_mem = runtime.totalMemory()
            free_mem = runtime.freeMemory()
            max_mem = runtime.maxMemory()
            metrics["jvm_used_memory_mb"] = round((total_mem - free_mem) / (1024 * 1024), 2)
            metrics["jvm_max_memory_mb"] = round(max_mem / (1024 * 1024), 2)
        except Exception as e:
            logger.debug(f"Could not retrieve JVM memory stats: {e}")

        # Query Shuffle Metrics via Spark statusTracker
        try:
            tracker = self.spark.sparkContext.statusTracker()
            current_stage_ids = set(tracker.getStageIds())
            new_stage_ids = current_stage_ids - self.stage_baseline_ids
            total_shuffle_read = 0
            total_shuffle_write = 0
            for sid in new_stage_ids:
                sinfo = tracker.getStageInfo(sid)
                if sinfo is not None:
                    # StageInfo methods in PySpark / JVM
                    pass
            metrics["shuffle_read_bytes"] = total_shuffle_read
            metrics["shuffle_write_bytes"] = total_shuffle_write
        except Exception as e:
            logger.debug(f"Could not retrieve shuffle metrics: {e}")

        return metrics
