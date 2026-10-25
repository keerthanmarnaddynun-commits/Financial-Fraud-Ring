"""
data_ingestion.py
Handles downloading/loading PaySim transaction data, synthetic fallback generation,
data cleaning, validation, and optimized Parquet persistence.
"""

import os
import sys
import zipfile
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType
)

from src.utils import get_spark_session, logger, SparkPerformanceTracker


PAYSIM_SCHEMA = StructType([
    StructField("step", IntegerType(), True),
    StructField("type", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("nameOrig", StringType(), True),
    StructField("oldbalanceOrg", DoubleType(), True),
    StructField("newbalanceOrig", DoubleType(), True),
    StructField("nameDest", StringType(), True),
    StructField("oldbalanceDest", DoubleType(), True),
    StructField("newbalanceDest", DoubleType(), True),
    StructField("isFraud", IntegerType(), True),
    StructField("isFlaggedFraud", IntegerType(), True)
])


def download_from_kaggle(dataset_name: str, target_dir: str) -> Optional[str]:
    """
    Attempts to download and extract the dataset using the Kaggle API.
    """
    logger.info(f"Attempting to download {dataset_name} using Kaggle API...")
    try:
        import kaggle
        kaggle.api.authenticate()
        os.makedirs(target_dir, exist_ok=True)
        kaggle.api.dataset_download_files(dataset_name, path=target_dir, unzip=True)
        logger.info(f"Successfully downloaded and extracted {dataset_name} into {target_dir}")
        for fname in os.listdir(target_dir):
            if fname.endswith(".csv"):
                return os.path.join(target_dir, fname)
    except Exception as e:
        logger.warning(f"Kaggle API download failed or credentials not configured: {e}")
    return None


def generate_synthetic_paysim(
    output_path: str,
    num_records: int = 50000,
    fraud_ratio: float = 0.03,
    random_seed: int = 42
) -> str:
    """
    Generates a high-fidelity synthetic PaySim financial transaction dataset with realistic
    fraud rings, layering chains, mule funneling networks, and balance dynamics.
    """
    logger.info(f"Generating realistic synthetic PaySim dataset with {num_records:,} records...")
    np.random.seed(random_seed)

    num_fraud = int(num_records * fraud_ratio)
    num_normal = num_records - num_fraud

    # 1. Generate Normal Transactions
    num_normal_users = max(500, int(num_normal * 0.35))
    num_merchants = max(100, int(num_normal * 0.05))

    normal_users = [f"C{100000000 + i}" for i in range(num_normal_users)]
    merchants = [f"M{200000000 + i}" for i in range(num_merchants)]

    txn_types = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
    type_probs = [0.40, 0.15, 0.25, 0.15, 0.05]

    normal_type = np.random.choice(txn_types, size=num_normal, p=type_probs)
    normal_step = np.random.randint(1, 744, size=num_normal)
    normal_amount = np.round(np.random.exponential(scale=2500.0, size=num_normal) + 5.0, 2)

    normal_orig = np.random.choice(normal_users, size=num_normal)
    # Payments go to merchants; others to normal users
    dest_candidates = np.array(normal_users + merchants)
    normal_dest = np.empty(num_normal, dtype=object)
    payment_mask = (normal_type == "PAYMENT")
    normal_dest[payment_mask] = np.random.choice(merchants, size=int(np.sum(payment_mask)))
    normal_dest[~payment_mask] = np.random.choice(normal_users, size=int(np.sum(~payment_mask)))

    normal_old_orig = np.round(normal_amount * np.random.uniform(1.05, 5.0, size=num_normal), 2)
    normal_new_orig = np.round(np.maximum(0.0, normal_old_orig - normal_amount), 2)
    normal_old_dest = np.round(np.random.exponential(scale=10000.0, size=num_normal), 2)
    normal_new_dest = np.round(normal_old_dest + normal_amount, 2)

    df_normal = pd.DataFrame({
        "step": normal_step,
        "type": normal_type,
        "amount": normal_amount,
        "nameOrig": normal_orig,
        "oldbalanceOrg": normal_old_orig,
        "newbalanceOrig": normal_new_orig,
        "nameDest": normal_dest,
        "oldbalanceDest": normal_old_dest,
        "newbalanceDest": normal_new_dest,
        "isFraud": 0,
        "isFlaggedFraud": 0
    })

    # 2. Generate Fraud Rings (Layering, Funneling, Smurfing)
    num_rings = 12
    ring_members = []
    for r in range(num_rings):
        ring_size = np.random.randint(5, 12)
        members = [f"C99900{r:02d}{m:02d}" for m in range(ring_size)]
        ring_members.append(members)

    fraud_records = []
    fraud_generated = 0

    while fraud_generated < num_fraud:
        ring = ring_members[np.random.randint(0, num_rings)]
        pattern = np.random.choice(["chain", "funnel", "cycle"])
        base_step = np.random.randint(10, 700)
        base_amt = round(float(np.random.uniform(50000.0, 450000.0)), 2)

        if pattern == "chain" and len(ring) >= 3:
            # A -> B -> C (Layering chain)
            chain_len = min(len(ring), np.random.randint(3, 6))
            sub = ring[:chain_len]
            for i in range(chain_len - 1):
                if fraud_generated >= num_fraud:
                    break
                s_acc, d_acc = sub[i], sub[i + 1]
                t_type = "TRANSFER" if i < chain_len - 2 else "CASH_OUT"
                amt = round(base_amt * (0.95 ** i), 2)
                fraud_records.append({
                    "step": base_step + i,
                    "type": t_type,
                    "amount": amt,
                    "nameOrig": s_acc,
                    "oldbalanceOrg": amt,
                    "newbalanceOrig": 0.0,
                    "nameDest": d_acc,
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": amt,
                    "isFraud": 1,
                    "isFlaggedFraud": 1 if amt > 200000 else 0
                })
                fraud_generated += 1

        elif pattern == "funnel" and len(ring) >= 4:
            # Multiple sources -> Mule Hub -> Cash Out
            hub = ring[-1]
            sources = ring[:-1]
            split_amt = round(base_amt / len(sources), 2)
            for s_acc in sources:
                if fraud_generated >= num_fraud:
                    break
                fraud_records.append({
                    "step": base_step,
                    "type": "TRANSFER",
                    "amount": split_amt,
                    "nameOrig": s_acc,
                    "oldbalanceOrg": split_amt,
                    "newbalanceOrig": 0.0,
                    "nameDest": hub,
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": split_amt,
                    "isFraud": 1,
                    "isFlaggedFraud": 0
                })
                fraud_generated += 1
            if fraud_generated < num_fraud:
                cashout_dest = f"C{np.random.randint(300000000, 399999999)}"
                fraud_records.append({
                    "step": base_step + 1,
                    "type": "CASH_OUT",
                    "amount": base_amt,
                    "nameOrig": hub,
                    "oldbalanceOrg": base_amt,
                    "newbalanceOrig": 0.0,
                    "nameDest": cashout_dest,
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": base_amt,
                    "isFraud": 1,
                    "isFlaggedFraud": 1 if base_amt > 200000 else 0
                })
                fraud_generated += 1

        else:
            # Cycle / Triangle dispersion
            if len(ring) >= 3:
                a, b, c = ring[0], ring[1], ring[2]
                for (src, dst) in [(a, b), (b, c), (c, a)]:
                    if fraud_generated >= num_fraud:
                        break
                    fraud_records.append({
                        "step": base_step,
                        "type": "TRANSFER",
                        "amount": base_amt,
                        "nameOrig": src,
                        "oldbalanceOrg": base_amt,
                        "newbalanceOrig": 0.0,
                        "nameDest": dst,
                        "oldbalanceDest": 0.0,
                        "newbalanceDest": base_amt,
                        "isFraud": 1,
                        "isFlaggedFraud": 1 if base_amt > 200000 else 0
                    })
                    fraud_generated += 1

    df_fraud = pd.DataFrame(fraud_records)
    df_combined = pd.concat([df_normal, df_fraud], ignore_index=True)
    df_combined = df_combined.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_combined.to_csv(output_path, index=False)
    logger.info(
        f"Synthetic PaySim dataset saved to {output_path} "
        f"({len(df_combined):,} rows, {df_combined['isFraud'].sum():,} frauds)."
    )
    return output_path


def load_and_clean_data(
    spark: SparkSession,
    raw_csv_path: str,
    output_parquet_path: str
) -> Tuple[int, str]:
    """
    Cleans raw transaction data, imputes missing values, engineers balance discrepancy features,
    and saves optimized partitioned Parquet files.
    """
    tracker = SparkPerformanceTracker(spark).start()
    logger.info(f"Loading raw transaction data from {raw_csv_path}...")

    df_raw = (
        spark.read
        .option("header", "true")
        .schema(PAYSIM_SCHEMA)
        .csv(raw_csv_path)
    )

    initial_count = df_raw.count()
    logger.info(f"Initial raw record count: {initial_count:,}")

    # Data Cleaning and Missing Value Handling
    df_clean = (
        df_raw
        .dropna(subset=["nameOrig", "nameDest", "amount", "step", "type"])
        .filter((F.col("amount") >= 0.0) & (F.col("step") > 0))
        .fillna({
            "oldbalanceOrg": 0.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFraud": 0,
            "isFlaggedFraud": 0
        })
    )

    # Feature Engineering: Accounting Balance Inconsistencies
    df_featured = (
        df_clean
        .withColumn("errorBalanceOrig", F.col("newbalanceOrig") + F.col("amount") - F.col("oldbalanceOrg"))
        .withColumn("errorBalanceDest", F.col("oldbalanceDest") + F.col("amount") - F.col("newbalanceDest"))
        .withColumn("isOrigZeroBalance", F.when(F.col("oldbalanceOrg") == 0.0, 1).otherwise(0))
        .withColumn("isDestZeroBalance", F.when(F.col("newbalanceDest") == 0.0, 1).otherwise(0))
    )

    # Save as optimized Parquet
    logger.info(f"Writing cleaned data to Parquet at {output_parquet_path}...")
    (
        df_featured
        .repartition(4)
        .write
        .mode("overwrite")
        .parquet(output_parquet_path)
    )

    metrics = tracker.stop()
    cleaned_count = df_featured.count()
    logger.info(
        f"Data cleaning and ingestion completed in {metrics['duration_seconds']}s. "
        f"Cleaned records: {cleaned_count:,} (Saved to {output_parquet_path})"
    )
    return cleaned_count, output_parquet_path


def run_ingestion(
    spark: SparkSession,
    data_dir: str = "data",
    num_synthetic_records: int = 50000,
    force_synthetic: bool = False
) -> str:
    """
    Orchestrates the entire ingestion stage: download or synthetic generation + cleaning.
    """
    raw_dir = os.path.join(data_dir, "raw")
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    csv_path = None
    if not force_synthetic:
        # Check if existing raw csv is present
        for fname in os.listdir(raw_dir):
            if fname.endswith(".csv"):
                csv_path = os.path.join(raw_dir, fname)
                logger.info(f"Found existing raw CSV: {csv_path}")
                break

        # If not present, try Kaggle API
        if not csv_path:
            csv_path = download_from_kaggle("ealaxi/paysim1", raw_dir)

    # Fallback to high-fidelity synthetic PaySim dataset
    if not csv_path:
        csv_path = os.path.join(raw_dir, "paysim_synthetic.csv")
        generate_synthetic_paysim(csv_path, num_records=num_synthetic_records)

    output_parquet = os.path.join(processed_dir, "transactions_clean.parquet")
    _, clean_path = load_and_clean_data(spark, csv_path, output_parquet)
    return clean_path


if __name__ == "__main__":
    spark = get_spark_session("DataIngestionStandalone")
    try:
        path = run_ingestion(spark, force_synthetic=True)
        print(f"Data ingestion completed successfully: {path}")
    finally:
        spark.stop()
