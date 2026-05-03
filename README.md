# Bank Marketing Campaign — Classifier Comparison

Predicting whether a client will subscribe to a term deposit based on a Portuguese bank's direct marketing campaign data. This project benchmarks multiple classification algorithms and identifies the best-performing model for deployment.

## Business Problem

Banks run costly phone-based marketing campaigns. Accurately predicting which clients are likely to subscribe to a term deposit allows the marketing team to:
- Prioritize high-value leads and reduce call volume
- Allocate resources more efficiently
- Increase conversion rates while lowering cost-per-acquisition

## Dataset

| Attribute | Value |
|---|---|
| Source | UCI Machine Learning Repository |
| Records | 41,188 clients |
| Features | 20 (demographic, campaign, economic) |
| Target | Binary — subscribed to term deposit (yes/no) |
| Class imbalance | ~11.3% positive class |

Key features: age, job, marital status, education, number of contacts, previous campaign outcome, economic indicators (employment rate, consumer price index).

## Models Evaluated

| Model | Train Accuracy | Test Accuracy | Train Time |
|---|---|---|---|
| Dummy Classifier (baseline) | 88.74% | 88.74% | — |
| Logistic Regression | **90.36%** | **90.06%** | Fast |
| Decision Tree | 91.82% | 85.21% | Fast |
| K-Nearest Neighbors | 90.91% | 88.12% | Moderate |
| SVM | 90.59% | 90.01% | Slow |

**Winner: Logistic Regression** — best generalization, no overfitting, interpretable coefficients.

## Methodology

- **EDA** — Distribution analysis, correlation heatmaps, class imbalance assessment
- **Preprocessing** — One-hot encoding for categoricals, StandardScaler for numerics, train/test split (80/20)
- **Model selection** — GridSearchCV with cross-validation for hyperparameter tuning
- **Evaluation** — Accuracy, precision, recall, F1-score, confusion matrix

## Key Findings

1. **Previous campaign outcome** is the strongest predictor — clients who subscribed before are far more likely to do so again
2. **Economic context matters** — employment rate and consumer confidence significantly influence subscription likelihood
3. **Contact duration** is highly correlated with success, though it cannot be known before the call
4. Logistic Regression matches SVM performance at a fraction of the computational cost

## Project Structure

```
bank-marketing-classifier-comparison/
├── prompt_III.ipynb        # Full analysis notebook
├── data/
│   ├── bank-additional-full.csv   # Full dataset (41,188 records)
│   └── bank-additional.csv        # Sampled dataset (4,119 records)
└── CRISP-DM-BANK.pdf       # CRISP-DM methodology reference
```

## Quick Start

```bash
git clone https://github.com/BedirhanUlas/bank-marketing-classifier-comparison.git
cd bank-marketing-classifier-comparison
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook prompt_III.ipynb
```

## Tech Stack

`Python` · `scikit-learn` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn`

## Future Improvements

- Apply SMOTE or class weighting to address the 11.3% positive class imbalance
- Evaluate ensemble methods (Random Forest, XGBoost) — expected significant gains
- Build a FastAPI serving endpoint for real-time scoring
- Feature engineering: interaction terms between economic indicators

## License

MIT
