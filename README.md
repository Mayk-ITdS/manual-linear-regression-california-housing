#  Manual Linear Regression on California Housing Data

This project implements a complete, manual pipeline for predicting house prices in California — **without using any machine learning libraries** like `scikit-learn` or `PyTorch`.

The purpose is to deeply understand the fundamentals of learning algorithms by **rebuilding everything from scratch** using matrix operations and mathematical logic.

---

## Objectives

This repository demonstrates:

- Manual ETL pipeline and Z-score normalization
-  Exploratory Data Analysis (EDA)
- Custom rank computation for Spearman correlation

- Pearson & Spearman correlation matrices (fully implemented from scratch)

- Manual train/test splitting

- Linear regression via:

  - Matrix algebra (β = (XᵀX)^-1 Xᵀy)

   - (Implemented) Gradient descent

- Custom loss function and manual weight updates

- (Planned) Manually built single-layer neural network for regression


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
## Feature Engineering
- Custom feature transformations implemented:

- RoomDiff = AveRooms - AveBedrms

- EstimatedHouseholds = Population / AveOccup

- OccupancyScore = RoomDiff / AveOccup

- Removed highly correlated features to reduce noise (AveRooms, AveBedrms, Population, etc.)
##  Project Structure

```plaintext
manual-linear-regression-california-housing/
├── README.md                  # Project overview and documentation
├── regression.py              # Full ETL + correlation + linear model (matrix-based,manual)
├── src/
│   ├── regression_manual.py   # Custom regression via matrix algebra
│   └── regression_gd.py       # Manual gradient descent implementation 
├── plots/                     # Visualizations (correlation matrix, predictions, etc.)
├── notebooks/                 # (Optional) Experimental notebooks & visualizations
└── .gitignore                 # Ignored files and directories
```

##  ML Roadmap Table

| ML Path                         | Neural Network Path |
|---------------------------------|----------------|
| Manual linear regression        |   Single-layer perceptron (manual)            |
| Pearson & Spearman correlation  |  Activation functions: ReLU, Sigmoid |
| Gradient descent implementation |  Multi-layer perceptron (MLP) |
| Regularization (L1/L2)          |  Dropout, BatchNorm |
| PCA, RFE                        |  ML vs NN performance comparison |

## Status
| Component                                     | Status |
| --------------------------------------------- | ------ |
| Data loading & normalization (ETL)            |  Complete |
| Pearson correlation (feature-feature/target)  | Complete |
| Spearman correlation (feature-feature/target) | Complete |
| Linear regression (matrix algebra)            | Complete |
| Linear regression (gradient descent)          | Complete |
| Custom training loop, loss & update rule      | Complete |
| Feature engineering                           | Complete |
| Single-layer neural network                   | In Progress |
| Evaluation metrics (R², RMSE, MAE)            | To Do |
| Weight saving/loading                         | To Do |
| Test set evaluation                           | To Do |
---

Author  
**Michał Zieliński**  
[GitHub: @Mayk-ITdS](https://github.com/Mayk-ITdS)  
Email: majk.develop@gmail.com  <!-- (replace with your real one if you want) -->
