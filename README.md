# Mod5_AIMLPA3

# Module 17 Practical Application 3: Bank Marketing Dataset

This project focuses on applying machine learning techniques to the **Bank Marketing Dataset** to predict whether a client will subscribe to a term deposit (target variable `y`).

---

## Business Objective
The main business objective is to help the bank improve marketing efficiency by predicting which customers are likely to subscribe to a term deposit.  
This reduces wasted calls, improves customer targeting, and increases campaign effectiveness.

---

## Dataset Overview
The dataset contains both **categorical** and **numeric** features, including client information, contact history, and macroeconomic indicators.  
Target:  
- `y` → whether the client subscribed (`yes`/`no`).
- Highly imbalanced dataset

---

## Steps Performed

### 1. Data Checks
- Checked for missing values and duplicates.
- Inspected incorrect or impossible values (`pdays`, `previous`, `default`).
- Identified outliers in numeric columns (`age`, `campaign`, `duration`).
- Converted columns to correct datatypes.

### 2. Exploratory Data Analysis (EDA)
- **Univariate**: Histograms, density plots, summary stats.  
- **Bivariate**: Correlation heatmaps, scatterplots, boxplots.  
- **Multivariate**: Combined feature visualizations.  
- Observed imbalance in target (`y`: ~89% no, 11% yes).

### 3. Preprocessing
- **Encoding**: One-Hot Encoding for categorical, Label Encoding for target.  
- **Scaling**: StandardScaler for Logistic Regression/Decision Tree; MinMaxScaler for KNN/SVM.  
- **Imbalance Handling**: SMOTE considered.  
- **Feature Selection**: Used correlation and `SelectKBest` to retain most informative features.  
- **Outlier Handling**: Capped extreme values (e.g., age, campaign).

### 4. Modeling
Built baseline and advanced models:  
- Logistic Regression  
- KNN (K-Nearest Neighbors)  
- Decision Tree  
- SVM (Support Vector Machine)  
- Dummy Classifier (baseline)

### 5. Model Comparison
Each model was compared on:  
- Train time  
- Train accuracy  
- Test accuracy  
- Precision, Recall, F1-score  
- ROC-AUC

### 6. Hyperparameter Tuning
- Used **GridSearchCV** to tune hyperparameters:
  - Logistic Regression (regularization)  
  - KNN (n_neighbors, weights)  
  - Decision Tree (max_depth, min_samples_split)  
  - SVM (kernel, C, gamma)

---

## Results (Sample)
| Model               | Train Time (s) | Train Acc | Test Acc |
|----------------------|----------------|-----------|----------|
| Logistic Regression | 0.12           | 0.89      | 0.87     |
| KNN                 | 0.32           | 0.92      | 0.88     |
| Decision Tree       | 0.05           | 1.00      | 0.85     |
| SVM                 | 1.44           | 0.93      | 0.90     |
| Dummy (baseline)    | 0.01           | 0.88      | 0.88     |

- **Decision Tree** risked overfitting (train=100% accuracy).  
- **SVM** gave the best balance of accuracy and generalization.  
- **Logistic Regression** remained strong and interpretable.  

---

## Next Steps
- More feature engineering (binning, interaction terms).  
- Try ensemble methods (Random Forest, Gradient Boosting, XGBoost).  
- Focus on **recall** (reduce false negatives → fewer missed opportunities).  
- The combination of re-balancing, feature enhancement, hyperparameter optimization, and metric alignment will provide more actionable and reliable models for the bank’s marketing strategy.
- Finally, it will be important to go beyond accuracy and evaluate the models on metrics that better align with the business goal—namely, maximizing recall and F1-score for the “yes” class to minimize the risk of missing potential customers


---

## Project Structure
```
├── Module_17_Practical_Application_3.ipynb   # Main notebook
├── bank+marketing.csv                        # Dataset
├── README.md                                 # Project documentation
└── images/                                   # Plots & visualizations
```

---

## Key Learnings
- Data preprocessing (encoding, scaling, outlier handling) is crucial.  
- Always compare multiple models; accuracy is not enough—check precision, recall, F1.  
- Hyperparameter tuning significantly improves model performance.  
- Interpretability matters: Logistic Regression is simple but powerful, while SVM and Trees capture complexity.



