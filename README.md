
# 📊 Comparing Classifiers on Bank Marketing Data

## 🧠 Business Understanding

The goal of this project is to develop predictive models to identify whether a customer will subscribe to a term deposit, based on data collected from a series of telemarketing campaigns by a Portuguese banking institution. This information can help improve marketing efficiency, reduce campaign costs, and improve targeting strategies.

---

## 📁 Dataset Overview

- **Records**: 41,188  
- **Features**: 20 (excluding target)  
- **Target variable**: `y` — whether the client subscribed to a term deposit (`yes` / `no`)  
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

---

## 🧹 Preprocessing & Feature Engineering

- Removed `duration` feature due to data leakage.
- Converted categorical features using one-hot encoding.
- Treated `"unknown"` values as separate categories.
- Encoded target variable: `yes` → 1, `no` → 0
- Splitted data into train/test (80% / 20%)
- Identified class imbalance: ~11% positive class

---

## 🔧 Baseline Model

- **Model**: DummyClassifier (strategy = `'most_frequent'`)
- **Baseline Accuracy**: `0.8874`

---

## 🧪 Model Training (Default Settings)

| Model                | Train Accuracy | Test Accuracy |
|---------------------|----------------|----------------|
| Logistic Regression | 0.9001         | 0.9006         |
| Decision Tree       | 0.9000         | 0.9000         |
| K-Nearest Neighbors | 0.9000         | 0.8967         |

---

## 🔍 Model Tuning (GridSearchCV)

### Decision Tree
- Best Parameters:  
  `max_depth=10`, `min_samples_split=2`
- **F1 Macro Score**: 0.67

### K-Nearest Neighbors
- Best `n_neighbors`: 9
- **Test Accuracy**: 0.8967

---

## 📊 Visualizations

- Bar chart comparing classifier accuracy
- Classification reports per model
- Support counts for target classes
- Confusion matrix (optional)

---

## ✅ Key Findings

- Logistic Regression performed best overall, with high accuracy and low overfitting.
- Decision Trees offer interpretability but may struggle with class imbalance.
- KNN underperformed slightly and may benefit from feature scaling.

---

## 🚀 Recommendations

- Try ensemble models (Random Forest, XGBoost).
- Use SMOTE or class weights to improve performance on the minority class.
- Monitor **recall** and **F1-score** instead of only accuracy.
- Build an interpretable dashboard for marketing decision-makers.

---

## 📌 Next Steps

- Document findings for stakeholders.
- Explore time-based model performance (by campaign month).
- Consider cost-sensitive learning to prioritize positive class predictions.
