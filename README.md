#  Manual Linear Regression on California Housing Data

This project implements a complete, manual pipeline for predicting house prices in California — **without using any machine learning libraries** like `scikit-learn` or `PyTorch`.

The purpose is to deeply understand the fundamentals of learning algorithms by **rebuilding everything from scratch** using matrix operations and mathematical logic.

---

## Objectives

This repository demonstrates:

-  Manual ETL pipeline and Z-score normalization
-  Exploratory Data Analysis (EDA)
-  Pearson and Spearman correlation analysis (implemented from scratch)**
- Manual train/test splitting
-  Linear regression via matrix algebra (`β = (XᵀX)^-1 Xᵀy`)
- (Planned) Gradient descent–based regression
-  (Planned) Manually built single-layer neural network


## Dataset

The dataset is sourced from the `fetch_california_housing()` utility in `scikit-learn.datasets`. It includes:

- Median income (`MedInc`)
- House age (`HouseAge`)
- Average rooms and bedrooms per household (`AveRooms`, `AveBedrms`)
- Population, household size (`Population`, `AveOccup`)
- Latitude & longitude
- **Target**: Median house value, in units of \$100,000

It’s based on real 1990 California census data, and widely used in regression benchmarks.

---

##  Project Structure

```plaintext
manual-linear-regression-california-housing/
├── README.md                  # Project overview and documentation
├── regression.py              # Full ETL + correlation + linear model (matrix-based)
├── src/
│   ├── regression_manual.py   # Custom regression via matrix algebra
│   └── regression_gd.py       # (Planned) Manual gradient descent version
├── plots/                     # Visualizations (correlation matrix, predictions, etc.)
├── notebooks/                 # (Optional) Experimental notebooks
└── .gitignore                 # Ignored files and directories
```

##  ML Roadmap Table

| ML Path                         | Neural Network Path                      |
|---------------------------------|------------------------------------------|
|  Manual linear regression       |  Manual single-layer neuron             |
|  Pearson & Spearman correlation |  Activation functions: ReLU, Sigmoid    |
| Gradient descent (regression) |  MLP (2–3 layers)                       |
|  Regularization (L1/L2)       |  Dropout, BatchNorm                     |
|  PCA, RFE                      |  ML vs NN performance comparison        |

## Status

| Component                                     | Status |
|-----------------------------------------------|--------|
| Data loading and normalization (ETL)          | Done   |
| Pearson correlation (feature-feature/target)  | Done   |
| Spearman correlation (feature-feature/target) | Done   |
| Linear regression (matrix algebra)            | Done   |
| Regression via gradient descent               | Upcoming |
| Manual single-layer neuron                    | In progress |
