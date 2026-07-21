"""
etl_pipeline.py
---------------
Automated Data Engineering Pipeline:
1. Extracts raw CSVs and loads them into an indexed SQLite relational database.
2. Performs vectorized data transformations and feature engineering.
3. Applies automated data validation checks (Governance & Integrity).
4. Export clean dataset to data/processed/ for downstream modeling.
"""

import os
import sqlite3
import numpy as np
import pandas as pd


def extract_and_load_sqlite(db_path: str, demo_path: str, acad_path: str) -> None:
    """Extracts raw CSV data and loads into SQLite database tables."""
    print(" [1/4] Extracting raw data and creating SQLite database...")

    conn = sqlite3.connect(db_path)

    df_demo = pd.read_csv(demo_path)
    df_acad = pd.read_csv(acad_path)

    # Write tables to SQLite
    df_demo.to_sql("dim_demographics", conn, if_exists="replace", index=False)
    df_acad.to_sql("fact_academics", conn, if_exists="replace", index=False)

    # Create index on Primary Key for SQL query performance optimization
    cursor = conn.cursor()
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_demo_id ON dim_demographics (student_id);"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_acad_id ON fact_academics (student_id);"
    )

    conn.commit()
    conn.close()
    print("       Successfully loaded tables into SQLite with optimized indexing.")


def transform_data(db_path: str) -> pd.DataFrame:
    """Queries SQLite database using SQL join and performs feature engineering using vectorized Pandas."""
    print(" [2/4] Executing SQL join and engineering analytical features...")

    conn = sqlite3.connect(db_path)

    # SQL query joining dimension and fact tables
    query = """
        SELECT 
            d.student_id,
            d.ethnicity,
            d.enrollment_pathway,
            d.first_generation,
            d.pell_eligible,
            d.residency_status,
            a.credits_attempted,
            a.term_gpa,
            a.lms_logins_per_week,
            a.advisor_visits,
            a.financial_hold,
            a.at_risk
        FROM fact_academics a
        JOIN dim_demographics d ON a.student_id = d.student_id;
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # --- Vectorized Feature Engineering (Optimized Performance) ---
    # 1. Flag for GPA falling below academic probation threshold (< 2.0)
    df["probation_risk"] = (df["term_gpa"] < 2.0).astype(int)

    # 2. Indicator for low engagement in Canvas LMS (< 1 login/day average)
    df["low_lms_engagement"] = (df["lms_logins_per_week"] < 7).astype(int)

    # 3. Combined compounding vulnerability index (First-Gen + Pell + Financial Hold)
    df["vulnerability_index"] = (
        df["first_generation"] + df["pell_eligible"] + df["financial_hold"]
    )

    # 4. One-Hot Encoding for categorical variables for downstream ML readiness
    categorical_cols = ["ethnicity", "enrollment_pathway", "residency_status"]
    df_encoded = pd.get_dummies(
        df, columns=categorical_cols, drop_first=True, dtype=int
    )

    return df_encoded


def validate_data_quality(df: pd.DataFrame) -> None:
    """Data Quality Management: Validates schema, null counts, and value ranges."""
    print(" [3/4] Running automated Data Quality & Governance checks...")

    # Rule 1: No missing values allowed in production payload
    null_counts = df.isnull().sum().sum()
    assert null_counts == 0, f" Data Quality Error: Found {null_counts} null values!"

    # Rule 2: Term GPA must fall strictly within valid 0.0 - 4.0 bounds
    assert (
        df["term_gpa"].min() >= 0.0 and df["term_gpa"].max() <= 4.0
    ), " Data Quality Error: Term GPA outside valid range!"

    # Rule 3: Target variable must be binary (0 or 1)
    assert set(df["at_risk"].unique()).issubset(
        {0, 1}
    ), " Data Quality Error: Target variable 'at_risk' contains invalid values!"

    print("       All Data Quality Checks Passed (0 nulls, valid GPA bounds).")


def load_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """Saves processed dataset to file for modeling."""
    print(" [4/4] Exporting clean feature store...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(
        f"       ETL Complete! Processed dataset saved to: {output_path} ({df.shape[0]:,} rows, {df.shape[1]} columns)"
    )


def main():
    raw_dir = os.path.join("data", "raw")
    db_path = os.path.join("data", "student_analytics.db")
    processed_path = os.path.join(
        "data", "processed", "processed_student_features.csv"
    )

    demo_path = os.path.join(raw_dir, "dim_student_demographics.csv")
    acad_path = os.path.join(raw_dir, "fact_student_academics.csv")

    print("Starting Automated Data Pipeline...")
    extract_and_load_sqlite(db_path, demo_path, acad_path)
    df_processed = transform_data(db_path)
    validate_data_quality(df_processed)
    load_processed_data(df_processed, processed_path)


if __name__ == "__main__":
    main()