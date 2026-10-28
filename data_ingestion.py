"""PaySim download, Spark setup, and input validation."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)


DATASET_ID = "ealaxi/paysim1"
DATA_DIR = Path(__file__).resolve().parent / "data"
GRAPHFRAMES_PACKAGE = os.getenv(
    "GRAPHFRAMES_PACKAGE", "io.graphframes:graphframes-spark3_2.12:0.9.3"
)

PAYMENTS_SCHEMA = StructType(
    [
        StructField("step", LongType(), True),
        StructField("type", StringType(), True),
        StructField("amount", DoubleType(), True),
        StructField("nameOrig", StringType(), True),
        StructField("oldbalanceOrg", DoubleType(), True),
        StructField("newbalanceOrig", DoubleType(), True),
        StructField("nameDest", StringType(), True),
        StructField("oldbalanceDest", DoubleType(), True),
        StructField("newbalanceDest", DoubleType(), True),
        StructField("isFraud", IntegerType(), True),
        StructField("isFlaggedFraud", IntegerType(), True),
    ]
)

REQUIRED_COLUMNS = [field.name for field in PAYMENTS_SCHEMA.fields]
NUMERIC_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


def create_spark_session(app_name: str = "PaySimFraudAnalytics") -> SparkSession:
    """Start a local Spark session, loading the matching GraphFrames JVM package."""
    builder = (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.jars.packages", GRAPHFRAMES_PACKAGE)
        .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "8g"))
        .config(
            "spark.jars.excludes",
            "org.slf4j:slf4j-api,org.scala-lang:scala-library",
        )
        .config(
            "spark.default.parallelism",
            os.getenv("SPARK_DEFAULT_PARALLELISM", "48"),
        )
        .config(
            "spark.sql.shuffle.partitions",
            os.getenv("SPARK_SHUFFLE_PARTITIONS", "48"),
        )
        .config("spark.ui.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    )
    return builder.getOrCreate()


def ensure_dataset(data_dir: Path = DATA_DIR) -> Path:
    """Return the local PaySim CSV, downloading it with the Kaggle CLI if absent."""
    data_dir.mkdir(parents=True, exist_ok=True)
    existing_csvs = sorted(data_dir.glob("*.csv"))
    if existing_csvs:
        return existing_csvs[0]

    configured_executable = os.getenv("KAGGLE_EXECUTABLE")
    executable = shutil.which(configured_executable or "kaggle")
    command = (
        [executable]
        if executable
        else [sys.executable, "-m", "kaggle"]
    ) + [
        "datasets",
        "download",
        "-d",
        DATASET_ID,
        "-p",
        str(data_dir),
        "--unzip",
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            "PaySim CSV was not found in data/. Install the Kaggle Python package "
            "and configure Kaggle API credentials (or place the dataset CSV in data/)."
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "Kaggle could not download ealaxi/paysim1. Check Kaggle credentials, "
            "dataset access, and network connectivity."
        ) from error

    downloaded_csvs = sorted(data_dir.glob("*.csv"))
    if not downloaded_csvs:
        raise FileNotFoundError(f"Kaggle download completed but no CSV exists in {data_dir}")
    return downloaded_csvs[0]


def load_and_validate_data(
    spark: SparkSession, csv_path: Optional[Path] = None
) -> DataFrame:
    """Load the known PaySim schema, remove malformed rows, and add a stable run ID."""
    path = csv_path or ensure_dataset()
    header = spark.read.option("header", True).option("inferSchema", False).csv(str(path))
    missing = sorted(set(REQUIRED_COLUMNS) - set(header.columns))
    if missing:
        raise ValueError(f"PaySim input is missing required columns: {', '.join(missing)}")
    raw = (
        spark.read.option("header", True)
        .option("mode", "PERMISSIVE")
        .schema(PAYMENTS_SCHEMA)
        .csv(str(path))
    )

    valid = raw.dropna(subset=REQUIRED_COLUMNS)
    for column in NUMERIC_COLUMNS:
        valid = valid.filter(~F.isnan(F.col(column).cast("double")))
    valid = valid.filter(
        F.col("type").isin("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")
        & F.col("nameOrig").startswith("C")
        & F.col("nameDest").startswith("C")
        & F.col("amount").between(0, float("inf"))
        & F.col("isFraud").isin(0, 1)
    )
    if not valid.take(1):
        raise ValueError(f"No valid PaySim transactions remain after validation: {path}")
    return valid.withColumn("transaction_id", F.monotonically_increasing_id())
