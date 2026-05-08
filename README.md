# sensor-drift-ml

**VOC gas classification under sensor drift — multi-class ML with distribution shift analysis**

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?style=flat-square)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red?style=flat-square)](https://xgboost.readthedocs.io)
[![Status](https://img.shields.io/badge/status-complete-brightgreen?style=flat-square)]()

---

## Overview

This project develops and evaluates machine learning classifiers for identifying six volatile organic compound (VOC) gases from a 128-feature metal oxide sensor array. The central finding is not which model performed best in cross-validation — it is why all models degraded on the held-out test set, and what that means for deploying sensor-based ML systems in production.

**Gases classified:** Ethanol · Ethylene · Ammonia · Acetaldehyde · Acetone · Toluene

**Dataset:** 3,436 training samples · 1,716 test samples · 128 numeric sensor features  
**Source:** University research dataset (Western Sydney University)

---

## Key finding: sensor drift dominates model performance

All five models achieve ~98% balanced accuracy in cross-validation on training data, then drop to 35–50% on the test set. This is **not overfitting.**

Distribution shift analysis reveals the cause:

- **46 of 128 features** have mean shifts exceeding 0.5 standard deviations between train and test periods
- **PC1 class means shift by 7–9 units per gas class** between time periods
- The sensors themselves responded differently in the later recording period — a well-documented phenomenon in chemical sensor arrays called **sensor drift**

> This makes raw accuracy a misleading metric entirely. Macro F1 and balanced accuracy are used throughout.

---

## Results

| Model | Features | Test accuracy | Macro F1 | Notes |
|-------|----------|--------------|----------|-------|
| Logistic Regression | Lasso 16 | 0.50 | 0.43 | Strong linear baseline. Fails on Acetaldehyde and Toluene (F1 = 0) |
| k-NN | Lasso 16 | 0.45 | 0.44 | Captures non-linear structure. Toluene still misclassified. Sensitive to drift |
| Random Forest | Lasso 16 | 0.37 | 0.31 | Handles correlated features natively. Underperformed — affected by temporal scoring inconsistency |
| XGBoost | Lasso 16 | 0.35 | 0.27 | Sequential boosting suited to hard classes. Hurt by drift without temporal retraining |
| Neural Network (MLP) | Lasso 16 | 0.50 | 0.40 | Best overall class balance. **Recommended for deployment** |

> Cross-validation balanced accuracy: ~98% across all models. Test performance gap is attributable to sensor drift, not model overfitting.

---

## Part A — EDA & preprocessing

### Dataset characteristics

- 3,436 training samples · 1,716 test samples · 128 sensor features · 6 gas classes
- No missing values or duplicates in either split
- **Class imbalance:** Toluene has 79 training samples vs 965 for Ethylene — a 12× ratio. The test set is perfectly balanced at 286 per class. All models use `class_weight='balanced'` to correct for this.

### Feature analysis

**Redundancy:** Correlation analysis found 128 feature pairs with r > 0.975. PCA confirms that 13–16 components explain 95% of total variance, with PC1 alone capturing 41.5%. The 128 features are effectively 13–16 independent signals.

**Class structure:** K-means clustering (k=6) on standardised features reveals a fan-shaped structure in PCA space:
- Ethylene and Acetone form clean, well-separated clusters
- Ethanol, Ammonia, and Acetaldehyde overlap significantly in the centre
- Non-linear decision boundaries are required — this directly informed model selection

### Feature engineering

All models use **16 Lasso-selected features** (LassoCV, α = 0.019, 16 non-zero coefficients from 128).

`StandardScaler` was fitted on training data only and applied to test — no data leakage.

---

## Part B — Model development

Five models of increasing complexity were trained and evaluated:

1. **Logistic Regression** — linear baseline with L2 regularisation
2. **k-Nearest Neighbours** — non-linear, distance-based
3. **Random Forest** — ensemble, handles correlated features natively
4. **XGBoost** — gradient boosting, suited to class imbalance
5. **Neural Network (MLP)** — multi-layer perceptron, recommended for deployment

### Deployment recommendation

The **MLP Neural Network** is recommended for production deployment. For robust operation under sensor drift:

| Recommendation | Detail |
|----------------|--------|
| Feature standardisation | Use a rolling window of recent sensor readings, not historical training statistics |
| Model refresh | Retrain or fine-tune periodically as new labelled data becomes available |
| Monitoring | Track per-class recall for **Toluene** and **Acetaldehyde** — highest misclassification risk |
| KPIs | Use **macro F1** and **balanced accuracy** — raw accuracy is misleading given class imbalance |

---

## Part C — Relevant recent method: test-time training (TTT)

Test-time training (TTT) is a paradigm in which a model adapts part of its parameters during inference using losses derived from the current input, typically through self-supervision. The model adjusts to the test distribution on the fly rather than relying on fixed training-time statistics.

TTT is directly applicable here: since sensor drift causes systematic distribution shift between training and test periods, a TTT approach could allow the model to partially recalibrate to current sensor behaviour at inference time — without requiring new labelled data. This is a strong candidate for future work given the 46-feature drift observed in this dataset.

---

## Repository structure
