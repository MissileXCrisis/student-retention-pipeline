
# 🎓 End-to-End Student Retention & Success Analytics Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue)
![SQL](https://img.shields.io/badge/SQL-SQLite3-lightgrey)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![Tableau](https://img.shields.io/badge/BI-Tableau%20Public-E97627)

> **Live Interactive Dashboard:** [View Tableau Public Dashboard](https://public.tableau.com/app/profile/missael.flores/viz/SDSUEnd-to-EndStudentRetentionSuccessDashboard/Dashboard1?publish=yes
)

---

## 📌 Executive Summary
This project builds an automated, end-to-end data pipeline that ingests demographic and academic performance data for **40,000 synthetic SDSU students**, executes automated data validation and feature engineering, trains a machine learning model to predict student dropout risk, and serves actionable insights to university leadership and academic advisors.

### Key Analytical Takeaway
Exploratory Data Analysis and Machine Learning feature importance revealed that **financial holds** carry a stronger statistical correlation with student attrition ($\text{Importance} = 0.248$) than low academic performance alone. Targeted micro-grants and early financial counseling offer the highest ROI for improving student retention.

---

## 🏗️ System Architecture

```text
[ Raw Student Data ] ──► [ SQLite Warehouse ] ──► [ Automated Python ETL ] ──► [ Validation Checks ]
   (Demographics &         (Indexed Relational       (Vectorized Pandas           (Governance &
     Academics)                  Tables)               Transformation)             Data Quality)
                                                                                       │
                                                                                       ▼
[ Tableau Public ] ◄── [ Batch Scoring Engine ] ◄── [ ML Predictive Model ] ◄── [ Feature Store ]
 (Interactive BI)        (Risk Classification)        (Random Forest Classifier)     (Processed Data)

🛠️ Tech Stack & MethodsData Processing & ETL: Python (Pandas, NumPy), SQL (SQLite3, SQLAlchemy)Machine Learning & Modeling: Scikit-Learn (Random Forest, Stratified K-Fold, Joblib)Business Intelligence: Tableau PublicDataOps & Governance: Git, Conda, Automated Assertion Validation, PII Anonymization📈 Model Performance & EvaluationThe model prioritizes Recall to minimize False Negatives (unidentified at-risk students who might drop out without support).ROC-AUC Score: $0.9284$At-Risk Recall: $91.32\%$ (Correctly flags >9 out of 10 vulnerable students)At-Risk Precision: $74.67\%$Top 5 Predictive Risk Drivers:term_gpa ($33.8\%$ importance)financial_hold ($24.8\%$ importance)advisor_visits ($13.1\%$ importance)vulnerability_index ($9.4\%$ importance)first_generation ($9.0\%$ importance)🔒 Data Governance & FERPA ComplianceData Privacy: All records use non-sequential, synthetic student IDs (SDSU-XXXXXX) with no Personally Identifiable Information (PII).Data Quality Management: The ETL pipeline executes strict data quality assertions—halting execution if missing values, invalid GPA bounds ($<0.0$ or $>4.0$), or schema mismatches are detected.🚀 How to Reproduce LocallyClone the repository:Bashgit clone [https://github.com/MissileXCrisis/student-retention-pipeline.git](https://github.com/MissileXCrisis/student-retention-pipeline.git)
cd student-retention-pipeline
Set up environment:Bashconda create -n retention_env python=3.10 -y
conda activate retention_env
pip install -r requirements.txt
Run the pipeline end-to-end:Bashpython src/generate_mock_data.py   # Generate 40k student dataset
python src/etl_pipeline.py         # Ingest to SQL, transform, and validate
python src/train_model.py          # Train & evaluate ML model
python src/score_students.py        # Score population for Tableau payload
