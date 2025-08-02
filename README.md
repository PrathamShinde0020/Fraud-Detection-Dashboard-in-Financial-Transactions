# 🔐 Fraud Detection in Financial Transactions using Machine Learning

This project presents a complete machine learning pipeline to detect fraudulent financial transactions based on user behavior, transaction context, and metadata. It applies supervised learning techniques on a structured dataset of 50,000 records and demonstrates how to build a high-recall model for identifying fraud with precision and interpretability.

📌 Project Objective

To develop a lightweight yet effective fraud detection system capable of:
- Identifying fraudulent transactions based on behavioral and contextual patterns.
- Handling class imbalance challenges common in fraud datasets.
- Explaining which features contribute most to fraudulent behavior.
- Laying the foundation for real-time fraud flagging and alerting.

🧠 Key Features

- ✅ Cleaned and engineered features from timestamp and behavioral columns.
- ✅ Encoded categorical variables and scaled numeric features.
- ✅ Applied class imbalance handling techniques (SMOTE & class weights).
- ✅ Trained interpretable models: **Logistic Regression** and **Random Forest**.
- ✅ Evaluated with **F1-score**, **ROC-AUC**, **Confusion Matrix**, and **Recall**.
- ✅ Extracted **feature importance** to understand key fraud indicators.
- ✅ Delivered business-focused insights on user patterns and high-risk behavior.

🛠️ Tech Stack

| Tool            | Purpose                                |
|-----------------|----------------------------------------|
| Python          | Core programming language              |
| pandas, numpy   | Data manipulation and preprocessing    |
| seaborn, matplotlib | Data visualization               |
| scikit-learn    | Machine learning modeling & metrics    |
| imbalanced-learn| Class imbalance handling (SMOTE)       |

📊 Model Workflow

1. Preprocessing
- Extracted `hour`, `day`, and `weekend_flag` from timestamps.
- Label encoded and one-hot encoded categorical fields.
- Normalized numerical features for model convergence.

2. Class Imbalance Handling
- Applied **SMOTE** to synthetically oversample the fraud class.
- Used `class_weight='balanced'` for model penalization.

3. Modeling
- Trained and compared:
  - **Logistic Regression** – Baseline performance
  - **Random Forest** – Final model (best balance of accuracy + explainability)

4. Evaluation
- **Precision, Recall, F1-score**: Focus on reducing false negatives.
- **ROC-AUC** and **Confusion Matrix**: Measured overall classification power.
- **Feature Importance**: Interpreted top contributors like Risk Score, Transaction Distance, Failed Attempts, etc.

🔍 Sample Insights

- Most frauds occurred during **non-business hours** and **weekends**.
- High fraud probability linked to **high transaction distances**, **failed authentication attempts**, and **unusual device usage**.
- Fraudulent users often showed **repeated failed transactions in short intervals**

📈 Next Steps (Optional Enhancements)
- Deploy model in a real-time dashboard using **Streamlit** or **Flask**.
- Integrate into a rule-based alerting system or banking backend.
- Fine-tune using **CatBoost** or **XGBoost** if real-time latency allows.

