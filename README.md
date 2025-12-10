# Titanic Survival Prediction - A Feature Engineering Project

**Student:** Arthur Gislon Leonida

This project demonstrates comprehensive **feature engineering** techniques to improve the prediction of passenger survival on the RMS Titanic. The work showcases advanced data manipulation, custom scikit-learn transformers, and rigorous model evaluation methodologies.

## 📊 Project Overview

This project implements a complete machine learning pipeline comparing baseline models against engineered feature models to demonstrate the impact of thoughtful feature engineering on predictive accuracy.

### Key Features

- **Custom Sklearn Transformers**: Built reusable pipeline components for feature extraction, binning, and encoding
- **From-Scratch Implementations**: Mutual Information (MI) and HSIC (Hilbert-Schmidt Independence Criterion) implemented without sklearn wrappers
- **Advanced Feature Selection**: Voting ensemble strategy combining MI and HSIC with cross-validation
- **Hyperparameter Optimization**: Optuna-based tuning for fair model comparison
- **Model Interpretability**: SHAP analysis for feature importance verification

## 🎯 Results

| Model Configuration | Accuracy | Improvement |
|:-------------------|:--------:|:-----------:|
| Baseline (Untuned) | 0.7263 | - |
| Baseline Tuned | 0.8045 | +10.8% |
| Engineered (Untuned) | 0.7765 | +6.9% |
| Engineered Tuned | 0.8045 | +10.8% |

**Key Finding**: Feature engineering provided a **6.9% improvement** over the untuned baseline, demonstrating that well-crafted features capture patterns that raw data cannot easily expose.

## 🛠️ Technical Implementation

### Feature Engineering Pipeline

1. **Feature Extraction**
   - Title extraction from passenger names (Mr, Miss, Mrs, Master, Other)
   - Cabin deck extraction and cabin counting
   - Family size creation (interaction feature)

2. **Optimal Binning**
   - Adaptive binning using Freedman-Diaconis and Knuth's Rule
   - Dynamic bin size determination based on data distribution
   - Quantile-based bin edges for robustness

3. **Transformations**
   - Box-Cox transformation for continuous features (Fare)
   - Log1p transformation for count features (SibSp, Parch, CabinCount)
   - One-Hot Encoding for nominal variables
   - IQR-based outlier capping

4. **Feature Selection**
   - Mutual Information (MI) scoring
   - HSIC (Hilbert-Schmidt Independence Criterion) scoring
   - Voting strategy: Intersection and Union of top-k features
   - 5-fold cross-validation for optimal feature set selection

### Files Structure

```
.
├── titanic_workflow.ipynb      # Main analysis notebook (22 sections)
├── helper_functions.py         # Custom functions (MI, HSIC, binning)
├── data/
│   └── train.csv              # Titanic dataset
└── README.md                  # This file
```

## 📚 Assignment Requirements

This project satisfies all assignment requirements:

- ✅ **Tabular Data**: Titanic dataset (classification)
- ✅ **Transformation/Interaction**: Box-Cox, Log1p, FamilySize interaction
- ✅ **Multiple Correlation Testing**: MI and HSIC with voting/ensemble strategy
- ✅ **Optimal Binning**: Adaptive binning with bins as variables
- ✅ **Top-k with SHAP Verification**: Iterative feature selection verified with SHAP
- ✅ **Comparison**: Tested with/without engineered features
- ✅ **From-Scratch Implementation**: MI and HSIC implemented manually in `helper_functions.py`

## 🔍 Key Insights

1. **Feature Engineering Impact**: The 6.9% improvement demonstrates that domain knowledge and feature extraction capture survival patterns (e.g., titles indicating gender/age) that raw features don't expose directly.

2. **One-Hot Encoding Matters**: Switching from label encoding to one-hot encoding for nominal variables was crucial for tree-based models to properly isolate categories.

3. **Hyperparameter Tuning Ceiling**: Both tuned models reached 0.8045, suggesting this is the performance ceiling for this specific train-validation split.

4. **Interpretability**: Engineered features (Title_Mr, Title_Mrs) are more interpretable than raw features, making the model's decisions easier to explain.

## 📦 Dependencies

Key libraries used:
- pandas, numpy: Data manipulation
- scikit-learn: ML pipeline and transformers
- XGBoost: Gradient boosting classifier
- Optuna: Hyperparameter optimization
- SHAP: Model interpretability
- scipy: Statistical transformations (Box-Cox)
- matplotlib, seaborn: Visualization

## 👨‍💻 Author

**Arthur Gislon Leonida**

This project demonstrates advanced feature engineering techniques, custom transformer implementation, and rigorous model evaluation for machine learning applications.