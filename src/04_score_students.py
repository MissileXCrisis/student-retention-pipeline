"""
score_students.py
-----------------
1. Loads trained Random Forest model artifact.
2. Generates continuous dropout probabilities (0.0 - 1.0) for all students.
3. Classifies students into actionable Risk Tiers (High, Medium, Low).
4. Exports a clean, human-readable CSV tailored for BI tools (Tableau / AWS QuickSight).
"""

import os
import sqlite3
import joblib
import pandas as pd


def main():
    print(" [1/3] Loading trained model artifact and feature store...")
    model_path = os.path.join("models", "retention_model.joblib")
    db_path = os.path.join("data", "student_analytics.db")
    processed_path = os.path.join(
        "data", "processed", "processed_student_features.csv"
    )

    # Load model and features
    model = joblib.load(model_path)
    df_features = pd.read_csv(processed_path)

    # Separate feature matrix matching training features
    X = df_features.drop(columns=["student_id", "at_risk"])

    print(" [2/3] Scoring student population & calculating probabilities...")
    # Generate continuous probabilities for Class 1 (At-Risk)
    risk_probabilities = model.predict_proba(X)[:, 1]

    # Query SQLite database to retrieve un-encoded, human-readable demographic fields
    conn = sqlite3.connect(db_path)
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
            a.at_risk AS actual_at_risk
        FROM fact_academics a
        JOIN dim_demographics d ON a.student_id = d.student_id;
    """
    df_dashboard = pd.read_sql_query(query, conn)
    conn.close()

    # Attach model outputs
    df_dashboard["risk_probability"] = risk_probabilities.round(4)

    # Bin probabilities into clear, actionable risk tiers for academic advisors
    # High Risk: Prob >= 0.70 | Medium Risk: 0.40 <= Prob < 0.70 | Low Risk: Prob < 0.40
    df_dashboard["risk_tier"] = pd.cut(
        df_dashboard["risk_probability"],
        bins=[-0.01, 0.3999, 0.6999, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )

    print(" [3/3] Exporting dashboard payload...")
    output_path = os.path.join(
        "data", "processed", "dashboard_student_predictions.csv"
    )
    df_dashboard.to_csv(output_path, index=False)

    print(
        f"\nSuccess! Dashboard payload saved to: {output_path} ({df_dashboard.shape[0]:,} rows)"
    )
    print("\nCalculated Risk Tier Breakdown:")
    print(df_dashboard["risk_tier"].value_counts().to_string())


if __name__ == "__main__":
    main()