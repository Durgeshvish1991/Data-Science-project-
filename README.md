# Customer Transaction Prediction

## Problem Statement
A bank wants to identify which customers are likely to make a specific transaction in the future, regardless of the amount, so the marketing team can target the right customers instead of running broad, low-conversion campaigns.

## Dataset
- Source: Kaggle — [add your dataset link here]
- Anonymized bank customer data with 200+ numerical features
- Target variable: `target` (1 = will transact, 0 = will not)
- Highly imbalanced dataset (~90% negative class)

## Approach
1. Performed data cleaning and exploratory data analysis (EDA) to understand feature distributions and class imbalance.
2. Handled class imbalance using up-sampling on the minority class.
3. Trained and compared three models: Logistic Regression, Random Forest, and LightGBM.
4. Evaluated models using accuracy and ROC-AUC (better suited than plain accuracy for imbalanced data).
5. LightGBM gave the best result — ~89% accuracy with strong ROC-AUC.

## Key Insights
- LightGBM outperformed simpler models mainly because it handles imbalanced, high-dimensional tabular data better.
- Up-sampling the minority class was necessary — without it, models were biased toward predicting "no transaction" for almost everyone.
- These predictions can help the bank prioritize marketing spend on customers most likely to transact, improving campaign ROI.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, LightGBM, Matplotlib/Seaborn

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook "PRCP-1003-Customer Transaction Prediction.ipynb"
```
