# Naive Bayes Classification — NBA Player Longevity Prediction

**Author:** Lapele Kenneth  
**Program:** 3MTT AI & Machine Learning Fellowship  
**Project Type:** Binary Classification with Probabilistic Model

---

## Dataset

- **Source:** Engineered NBA Players dataset (output from Feature Engineering project)
- **Size:** 1,340 players x 10 features
- **File:** `nba_engineered.csv`
- **Task:** Predict whether an NBA player will last **5+ years** in the league

---

## Modeling Approach

A **Gaussian Naive Bayes** classifier was chosen as the primary model because:
- It is well-suited for continuous numeric features (all features here are continuous)
- It provides fast, interpretable probabilistic predictions
- It serves as a strong baseline before testing complex ensemble models
- Its class-prior and likelihood outputs map naturally to scouting probability scores

### Pipeline
1. Load engineered dataset and confirm `target_5yrs` as classification target
2. Split data 80/20 with stratification to preserve class balance
3. Apply StandardScaler to normalise feature ranges
4. Train GaussianNB and generate predictions on the held-out test set
5. Evaluate with Confusion Matrix, Precision, Recall, and F1 Score
6. Analyse feature importance via class-conditional mean differences
7. Critique the independence assumption in the basketball context

---

## Final Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 63.8% |
| Precision | 80.0% |
| Recall | 55.4% |
| F1 Score | 65.5% |

**Confusion Matrix (268 test players):**

|  | Predicted: <5yrs | Predicted: 5+yrs |
|--|--|--|
| **Actual: <5yrs** | 79 (TN) | 23 (FP) |
| **Actual: 5+yrs** | 74 (FN) | 92 (TP) |

---

## Top Predictive Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `total_points` | Highest class separator |
| 2 | `reb` | Strong playing time indicator |
| 3 | `tov` | Ball-handling volume proxy |
| 4 | `stl` | Defensive engagement |
| 5 | `fg` | Shooting efficiency |

---

## Independence Assumption Analysis

Gaussian Naive Bayes assumes all features are **conditionally independent** given the class. This assumption is **violated** in basketball data:
- `total_points` and `tov` are highly correlated (r > 0.85)
- `fg%` and `efficiency` overlap by construction
- `ast` and `tov` move together for playmakers

Despite this, the model achieves **80% precision**, making it a reliable first-pass scouting filter. Tree-based models (Random Forest, XGBoost) are recommended for final decisions.

---

## Repository Contents

```
naive_bayes_nba.ipynb              # Main notebook (all cells executed)
nba_engineered.csv                 # Input dataset
confusion_matrix.png               # Confusion matrix heatmap
model_metrics.png                  # Performance metrics bar chart
feature_importance.png             # Feature importance chart
class_distribution.png             # Train/test class balance
independence_violation_heatmap.png # Correlation heatmap (assumption analysis)
README.md                          # This file
```

---

## How to Run

```bash
git clone https://github.com/lapele/naive-bayes-nba.git
cd naive-bayes-nba
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook naive_bayes_nba.ipynb
```

---

## Tools Used

- **Python 3.10**
- **pandas, NumPy** — data handling
- **scikit-learn** — GaussianNB, StandardScaler, metrics
- **seaborn, matplotlib** — visualisation
- **SciPy** — statistical analysis
