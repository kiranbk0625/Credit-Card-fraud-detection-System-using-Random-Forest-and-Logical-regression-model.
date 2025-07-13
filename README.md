✅ 1. Documentation
📌 Project Title:
Credit Card Fraud Detection System

📌 Objective:
To develop a machine learning-based system that detects potentially fraudulent credit card transactions in real-time or post-transaction, improving financial security.

📌 Problem Statement:
Credit card fraud poses a significant challenge to financial institutions. Detecting fraudulent transactions from massive datasets while maintaining high accuracy and low false positives is a complex problem. Traditional rule-based systems lack adaptability to new fraud patterns.

📌 Proposed Solution:
Develop a predictive model using machine learning algorithms like Logistic Regression, Random Forest, or XGBoost that learns patterns from labeled transaction data and classifies transactions as fraudulent or legitimate.

📌 Scope:
Analyze transaction data

Handle class imbalance (fraud cases are rare)

Train multiple classification models

Evaluate performance with precision, recall, and F1-score

Use the best-performing model for deployment or testing

📌 Tools and Technologies:
Languages: Python

Libraries: pandas, numpy, scikit-learn, seaborn, matplotlib, imbalanced-learn

Environment: Jupyter Notebook, Google Colab or local IDE

Dataset: Public dataset from Kaggle - Credit Card Fraud Detection

📌 Key Features:
Data preprocessing & scaling

Handling class imbalance with SMOTE

Feature importance analysis

ROC-AUC and confusion matrix visualizations

Real-time or batch prediction pipeline

✅ 2. Flow of the Project
🔁 System Flow Diagram (Step-by-step):
java
Copy
Edit
Raw Dataset (CSV)
       ↓
Data Preprocessing
  - Null value check
  - Data normalization (StandardScaler)
  - Label separation (X & y)
       ↓
Train/Test Split (80:20)
       ↓
Class Imbalance Handling (SMOTE)
       ↓
Model Training (Logistic Regression, Random Forest, XGBoost)
       ↓
Model Evaluation (Confusion Matrix, ROC Curve, F1-Score)
       ↓
Fraud Prediction API / UI Integration
🔧 Modules:
Data Collection & Preprocessing Module
Load dataset, remove outliers, standardize features.

Modeling Module
Train models and select the best one based on evaluation metrics.

Evaluation Module
Precision, recall, F1-score, confusion matrix, ROC-AUC score.

Prediction Module
Predict fraud status of new transactions.
<img width="1181" height="807" alt="image" src="https://github.com/user-attachments/assets/202bc23e-5128-4d0f-9427-425cc3ef36f4" />


✅ 3. Model Used for Detection
🎯 Baseline Model: Logistic Regression
Simple and interpretable

Performs well with standardized data

🌲 Best Performing Model: Random Forest Classifier
Robust to overfitting

Handles imbalanced datasets better than most models

Provides feature importance

📊 Model Evaluation Metrics:
Accuracy: ~99%

Precision (Fraud Class): High precision avoids false alarms.

Recall: Important to catch most frauds.

F1-score: Balanced measure of precision and recall.

ROC-AUC Score: ~0.98
# install required libraries if not already installed
# !pip install kagglehub imbalanced-learn scikit-learn pandas matplotlib seaborn

import kagglehub
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Step 1: Download dataset from Kaggle
print("[INFO] Downloading dataset...")
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
csv_path = f"{path}/creditcard.csv"
print("Downloaded to:", csv_path)

# Step 2: Load the dataset
print("[INFO] Loading dataset...")
df = pd.read_csv(csv_path)
print(df.head())

# Step 3: Basic EDA (optional)
print("\nClass Distribution:\n", df['Class'].value_counts())

# Step 4: Preprocessing
X = df.drop(['Class', 'Time'], axis=1)
y = df['Class']

# Normalize 'Amount'
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X['Amount'].values.reshape(-1, 1))

# Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Step 6: Handle class imbalance
print("[INFO] Applying SMOTE...")
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

# Step 7: Train model
print("[INFO] Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_resampled, y_resampled)

# Step 8: Evaluate model
y_pred = model.predict(X_test)
print("\n[RESULT] Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n[RESULT] Classification Report:\n", classification_report(y_test, y_pred))
print("[RESULT] ROC AUC Score:", roc_auc_score(y_test, y_pred))

# Step 9: Feature Importance (optional)
importances = model.feature_importances_
features = X.columns
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.show()
