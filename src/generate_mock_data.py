"""
generate_mock_data.py
---------------------
Generates a realistic, synthetic 40,000-student dataset reflecting SDSU demographic 
distributions and academic performance indicators.

Outputs:
  - data/raw/dim_student_demographics.csv
  - data/raw/fact_student_academics.csv
"""

import os
import numpy as np
import pandas as pd

# Set seed for reproducibility across runs
np.random.seed(42)

NUM_STUDENTS = 40000


def generate_demographics(num_students: int) -> pd.DataFrame:
    """Generates student demographic dimension table with SDSU-aligned probabilities."""

    student_ids = [f"SDSU-{i:06d}" for i in range(1, num_students + 1)]

    # SDSU Demographic Distributions
    ethnicities = [
        "Hispanic / Latino",
        "White",
        "Asian",
        "Two or More Races",
        "Other / International",
        "Black / African American",
    ]
    ethnicity_probs = [0.36, 0.33, 0.13, 0.07, 0.075, 0.035]

    ethnicity = np.random.choice(ethnicities, size=num_students, p=ethnicity_probs)
    enrollment_pathway = np.random.choice(
        ["First-Time Freshman", "Transfer"], size=num_students, p=[0.65, 0.35]
    )
    first_gen = np.random.choice([1, 0], size=num_students, p=[0.40, 0.60])
    pell_eligible = np.random.choice([1, 0], size=num_students, p=[0.38, 0.62])
    residency = np.random.choice(
        ["In-State", "Out-of-State", "International"],
        size=num_students,
        p=[0.85, 0.11, 0.04],
    )

    df_demo = pd.DataFrame(
        {
            "student_id": student_ids,
            "ethnicity": ethnicity,
            "enrollment_pathway": enrollment_pathway,
            "first_generation": first_gen,
            "pell_eligible": pell_eligible,
            "residency_status": residency,
        }
    )

    return df_demo


def generate_academics(df_demo: pd.DataFrame) -> pd.DataFrame:
    """Generates academic performance and engagement fact table with embedded risk logic."""

    num_students = len(df_demo)

    # Base academic distributions
    credits_attempted = np.random.choice(
        [12, 13, 14, 15, 16, 17, 18],
        size=num_students,
        p=[0.10, 0.05, 0.15, 0.45, 0.15, 0.05, 0.05],
    )

    # Base GPA centered ~3.0, bounded [0.0, 4.0]
    base_gpa = np.random.normal(loc=3.0, scale=0.55, size=num_students)
    term_gpa = np.clip(np.round(base_gpa, 2), 0.0, 4.0)

    # LMS Logins/week (Canvas)
    lms_logins_per_week = np.random.poisson(lam=18, size=num_students)

    # Academic advising visits
    advisor_visits = np.random.choice(
        [0, 1, 2, 3, 4], size=num_students, p=[0.35, 0.35, 0.18, 0.08, 0.04]
    )

    # Financial Holds (0 = No Hold, 1 = Financial Hold)
    financial_hold = np.random.choice([0, 1], size=num_students, p=[0.88, 0.12])

    # Construct dataframe
    df_acad = pd.DataFrame(
        {
            "student_id": df_demo["student_id"],
            "credits_attempted": credits_attempted,
            "term_gpa": term_gpa,
            "lms_logins_per_week": lms_logins_per_week,
            "advisor_visits": advisor_visits,
            "financial_hold": financial_hold,
        }
    )

    # Injecting Attrition Risk Probability Signal (Logic for ML model to learn)
    # -----------------------------------------------------------------------
    # Risk factors: Low GPA, Low LMS activity, Financial holds, First-Gen with 0 advisor visits
    risk_score = (
        (df_acad["term_gpa"] < 2.25).astype(int) * 2.5
        + (df_acad["lms_logins_per_week"] < 7).astype(int) * 1.8
        + (df_acad["financial_hold"] == 1).astype(int) * 1.2
        + (
            (df_demo["first_generation"] == 1) & (df_acad["advisor_visits"] == 0)
        ).astype(int)
        * 1.0
        + np.random.normal(0, 0.5, size=num_students)  # Adds realistic noise
    )

    # Convert continuous risk score into binary target (1 = At-Risk / Dropped Out, 0 = Retained)
    # Threshold chosen to yield ~12-15% dropout/risk rate standard for higher ed
    risk_probability = 1 / (1 + np.exp(-risk_score))
    df_acad["at_risk"] = (risk_probability > 0.72).astype(int)

    return df_acad


def main():
    print("Generating synthetic SDSU student dataset...")

    # Ensure output directory exists
    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)

    # Generate tables
    df_demographics = generate_demographics(NUM_STUDENTS)
    df_academics = generate_academics(df_demographics)

    # File paths
    demo_path = os.path.join(output_dir, "dim_student_demographics.csv")
    acad_path = os.path.join(output_dir, "fact_student_academics.csv")

    # Save to CSV
    df_demographics.to_csv(demo_path, index=False)
    df_academics.to_csv(acad_path, index=False)

    print(f"Success! Generated {NUM_STUDENTS:,} records.")
    print(f"Demographics saved to: {demo_path}")
    print(f"Academics saved to:    {acad_path}")
    print(f"Overall At-Risk Rate:  {df_academics['at_risk'].mean():.2%}")


if __name__ == "__main__":
    main()