# Machine Learning Midterm Project
## Sahrul Ridho Firdaus - 1103223009

This repository contains three machine learning projects covering **Classification**, **Regression**, and **Clustering** tasks as part of the Machine Learning Midterm assignment.

---

## 📁 Repository Structure

```
midterm-machine-learning/
├── midterm-ML-1.ipynb              # Classification - Fraud Detection
├── midterm-ML-2 regresi.ipynb      # Regression - Target Prediction
├── midterm_ML_3_clustering.ipynb   # Clustering - Customer Segmentation
└── README.md                       # Project Documentation
```

---

## 📌 Project Overview

| Notebook | Task Type | Dataset | Model(s) Used |
|----------|-----------|---------|---------------|
| `midterm-ML-1.ipynb` | Classification | Fraud Transaction | Random Forest |
| `midterm-ML-2 regresi.ipynb` | Regression | Midterm Regression Dataset | LightGBM, CatBoost, XGBoost (Ensemble) |
| `midterm_ML_3_clustering.ipynb` | Clustering | Customer Credit Card Data | K-Means |

---

## 📓 Notebook Descriptions

### 1. Classification - Fraud Detection (`midterm-ML-1.ipynb`)

**Objective:** Build a machine learning model to detect fraudulent online transactions.

**Workflow:**
- Exploratory Data Analysis (EDA)
- Data preprocessing (handling missing values, encoding categorical features)
- Dropping columns with >80% missing values
- Label Encoding for categorical variables
- Train/Validation split with stratification
- Model training using Random Forest Classifier
- Hyperparameter tuning with RandomizedSearchCV

**Model Performance:**

| Metric | Score |
|--------|-------|
| Accuracy | 97.38% |
| Precision | 66.43% |
| Recall | 50.86% |
| F1-Score | 0.5761 |
| **ROC-AUC** | **0.9335** |

**Key Insights:**
- Excellent discriminative ability with ROC-AUC of 0.9335
- High precision minimizes false alarms
- Moderate recall indicates room for improvement in catching all fraud cases

---

### 2. Regression - Target Prediction (`midterm-ML-2 regresi.ipynb`)

**Objective:** Build an end-to-end regression model to predict a continuous target variable.

**Workflow:**
- Data loading and exploration
- Duplicate removal and outlier handling (IQR method)
- Feature scaling with StandardScaler/MinMaxScaler
- Dimensionality reduction using PCA (98% variance retained)
- Multiple model training: LightGBM, CatBoost, XGBoost
- Weighted ensemble for improved predictions

**Model Performance:**

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| LightGBM | - | - | Baseline |
| CatBoost | Improved | Improved | Higher |
| **Weighted Ensemble** | **Best** | **Best** | **Best** |

**Key Insights:**
- PCA used to reduce dimensionality while retaining 98% variance
- Ensemble of CatBoost, XGBoost, and LightGBM achieved best results
- Weighted averaging based on individual R² scores

---

### 3. Clustering - Customer Segmentation (`midterm_ML_3_clustering.ipynb`)

**Objective:** Segment customers based on their spending and payment behavior using unsupervised learning.

**Workflow:**
- Data loading from Google Drive
- Data preprocessing (drop CUST_ID, impute missing values with median)
- Outlier removal using IQR method
- Feature engineering (purchase_ratio, balance_ratio, pay_vs_limit, ca_amount_per_trx)
- Feature scaling with StandardScaler
- Elbow Method and Silhouette Score for optimal cluster selection
- K-Means clustering with k=4
- PCA visualization of clusters

**Cluster Analysis:**

| Cluster | Customer Profile | Risk Level |
|---------|------------------|------------|
| 0 | Moderate spenders, stable activity | Low Risk |
| 1 | High income, frequent spenders | Very Low Risk |
| 2 | Low-moderate income, payment delays | High Risk |
| 3 | New customers, minimal transactions | Unknown Risk |

**Key Insights:**
- 4 distinct customer segments identified
- Feature engineering improved cluster separation
- PCA visualization confirms clear cluster boundaries

---

## 🚀 How to Navigate

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hiraeth12/midterm-machine-learning.git
   ```

2. **Open in Google Colab or Jupyter:**
   - Each notebook can be opened directly in Google Colab using the badge at the top
   - Or run locally with Jupyter Notebook/Lab

3. **Dataset Setup:**
   - Classification: Dataset downloaded via the notebook
   - Regression: Requires `midterm-regresi-dataset.csv` in the working directory
   - Clustering: Dataset downloaded from Google Drive via `gdown`

4. **Dependencies:**
   ```
   pandas, numpy, matplotlib, seaborn, scikit-learn
   torch (for GPU check)
   catboost, xgboost, lightgbm (for regression)
   ```

---

## 📊 Summary of Results

| Task | Best Model | Key Metric | Score |
|------|------------|------------|-------|
| Classification | Random Forest (Tuned) | ROC-AUC | 0.9335 |
| Regression | Weighted Ensemble | R² | Best Performance |
| Clustering | K-Means (k=4) | Silhouette Score | Optimal at k=4 |

---

## 📝 License

This project is for educational purposes as part of the Machine Learning course midterm assignment.

---

*Generated for Machine Learning Midterm - 2024*
