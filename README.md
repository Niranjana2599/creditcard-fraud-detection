# 💳 Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using machine learning.  
The dataset is highly imbalanced, with fraud cases being extremely rare compared to genuine transactions.  
Therefore, special techniques such as **SMOTE oversampling**, **feature scaling**, and **threshold tuning** were used.

---

## 🔍 Problem Statement
Financial institutions must detect fraudulent transactions to prevent unauthorized charges.  
However, fraudulent transactions are very rare and often resemble legitimate ones, making detection challenging.

---

## 🧠 Approach

| Step | Description |
|-----|-------------|
| **1. Data Exploration** | Studied class imbalance, feature distributions, correlations |
| **2. Preprocessing** | Removed duplicates, scaled features, handled imbalance using SMOTE |
| **3. Model Training** | Trained ML models including Logistic Regression, Random Forest, XGBoost|
| **4. Evaluation** | Used Precision, Recall, F1 score, ROC-AUC, and AUPRC metrics |
| **5. Threshold Tuning** | Improved fraud detection by adjusting decision threshold |

---

## ⚙️ Techniques Used
- **SMOTE** for oversampling fraud cases
- **StandardScaler** for scaling transaction amount
- **Random Forest** for final prediction model
- **Threshold tuning** to improve recall on fraud cases

---

## 🏆 Best Performing Model: Random Forest
| Metric | Score |
|-------|--------|
| **Accuracy** | ~100% |
| **Precision (Fraud)** | **0.83** |
| **Recall (Fraud)** | **0.83** |
| **AUPRC** | **0.87** |
| **ROC-AUC** | **0.98+** |

✅ The model correctly detects majority of fraud transactions  
✅ Low false positives  
✅ Strong overall performance

---

