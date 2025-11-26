# Customer Churn Prediction

## Overview
This project uses machine learning to predict customer churn for a telecom provider. Churn prediction is critical in telecom because losing a customer is far more expensive than retaining one. By analyzing customer usage and account data, the model flags customers at risk of leaving so that targeted retention strategies can be applied.

## Dataset
The data is the **Telco Customer Churn** dataset (Kaggle), containing **7,043** customer records with **21** features. Features include demographic attributes (e.g. gender, SeniorCitizen, Partner, Dependents), service usage (InternetService, OnlineSecurity, OnlineBackup, etc.), and billing info (tenure, MonthlyCharges, TotalCharges, Contract type, PaymentMethod). The target variable is **Churn** (Yes/No).

## Key Features
- **Data Cleaning:** Handles missing values and encodes categorical fields.
- **Class Imbalance:** Applies SMOTE oversampling to balance the minority churn class (~27% of records).
- **Modeling:** Trains ensemble classifiers (Random Forest, XGBoost) for robust churn prediction.
- **Evaluation:** Reports accuracy and class-specific metrics (precision, recall, F1) to assess performance.

## Technologies Used
- **Languages/Libraries:** Python, pandas, scikit-learn, XGBoost, imbalanced-learn (SMOTE), Matplotlib
- **Environment:** Jupyter Notebook

