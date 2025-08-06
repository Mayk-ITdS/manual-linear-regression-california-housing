
# Custom Linear Regression Engine Built From Scratch (No ML Libraries)

# Overview

This project implements a from-scratch linear regression engine to predict housing prices in California using the 1990 U.S. Census dataset. It avoids machine learning libraries like scikit-learn, PyTorch, or TensorFlow for training, focusing instead on low-level mathematical foundations such as:

- Feature engineering & rank transformation

- Z-score & Spearman normalization

- Gradient descent optimization

- Correlation matrix analysis

- Custom ETL pipeline and model visualization

## How to run


### 1. Train model from scratch
```
from regression import linear_regresion, etl, features_engineering, df_full

etl_data = etl(features_engineering(df_full.copy()))
df_result, df_result_sorted, model = linear_regresion(etl_data)

#It will generate a model.json if it`s not there
``` 


### 2. Load saved model
```
from regression import load_model

model = load_model("model.json")  # Default path
```
### 3. Generate visualizations

```
from visualiser import Visualiser

v = Visualiser(X_zscore, Y_zscore, df_xy_corr, feature_names, df_result)
v.correlation_plot()
v.prediction_vs_target()
v.one_choice()
```
### Example Plot

<img src="./plots/pred_vs_target.png" alt="Prediction vs Target" width="600">

##  ML Roadmap Table

| ML Path                         | Neural Network Path |
|---------------------------------|----------------|
| Manual linear regression        |   Single-layer perceptron (manual)            |
| Pearson & Spearman correlation  |  Activation functions: ReLU, Sigmoid |
| Gradient descent implementation |  Multi-layer perceptron (MLP) |
| Regularization (L1/L2)          |  Dropout, BatchNorm |
| PCA, RFE                        |  ML vs NN performance comparison |

## Status
| Component                                     | Status          |
| --------------------------------------------- |-----------------|
| Data loading & normalization (ETL)            | Complete        |
| Pearson correlation (feature-feature/target)  | Complete        |
| Spearman correlation (feature-feature/target) | Complete        |
| Linear regression (matrix algebra)            | Complete        |
| Linear regression (gradient descent)          | Complete        |
| Custom training loop, loss & update rule      | Complete        |
| Feature engineering                           | Complete        |
| Single-layer neural network                   | **In Progress** |
| Evaluation metrics (R², RMSE, MAE)            | **To Do**       |
| Weight saving/loading                         | Complete        |
| Test set evaluation                           | **To Do**       |
---



## Motivation 
This repository is part of a broader journey toward mastering:

Data Science → Machine Learning → Deep Learning → Neural Networks → AI Systems Engineering

The project’s emphasis is on solidifying fundamentals, writing clean mathematical code, and treating every decision as an engineer would – with clarity and traceability.


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

This project uses the California Housing dataset from sklearn.datasets.fetch_california_housing(). 

It includes demographic and geographic features from the 1990 U.S. Census, with the target being median home value per block group (in $100,000s).

- Median income (`MedInc`)
- House age (`HouseAge`)
- Average rooms and bedrooms per household (`AveRooms`, `AveBedrms`)
- Population, household size (`Population`, `AveOccup`)
- Latitude & longitude
- **Target**: Median house value, in units of $100,000
---
## Feature Engineering
Custom feature transformations implemented:

- RoomDiff = AveRooms - AveBedrms:  **attempts to quantify unused room space**

- EstimatedHouseholds = Population / AveOccup: **infers household count**

- OccupancyScore = RoomDiff / AveOccup: **scales living space per occupant**

Removed highly correlated features to reduce noise (AveRooms, AveBedrms, Population, etc.)

## Features and Techniques

- Linear Regression via manual gradient descent

- Feature engineering: new domain-specific fields added

- Spearman and Pearson correlation calculations

- Z-score standardization + Spearman rank normalization

- Training loop with early stopping

- Model persistence in JSON (manual loading + inference)

- Modular visualization layer for predictions and errors

## model.json contents

```
{
  "weights": [...],
  "bias": ...,
  "mean_target": ...,
  "std_target": ...,
  "mean_features": [...],
  "std_features": [...],
  "real_pred": [...]
}
```
### Example Results

| Metric                 | Value     | Description                                |
| ---------------------- | --------- | ------------------------------------------ |
| **Best MSE (z-score)** | \~0.3525  | Average squared error on standardized data |
| **RMSE**               | \~0.59    | Interpreted as \~\$59,000 prediction error |
| **Absolute Error**     | \~1.5–2.0 | Mostly within \$150k–200k from truth       |
| **Epochs**             | \~60–120  | Training converges early                   |
| **Stability**          | Moderate  | Low variance in training, some outliers    |


##  Project Structure

```
Linear-regression-housing/
├── src/
│   └── linear_regression/
│       ├── regression.py         # Core training logic, model serialization
│       ├── visualizer.py         # Plotting class (modular design)
│       ├── model.json            # Saved model state (weights, stats)
│       └── test.py               # Script for analysis and plots
└── README.md                     
                
```


### Next Steps
Integrate sklearn’s regression for baseline comparison

- Add train/validation/test split with metrics per set

- Extend model to polynomial regression

- Build dashboard to visualize live predictions

- Package as CLI or web app

- Add unit tests for training loop and ETL

- Switch to np.linalg.lstsq() for closed-form solution comparison




Created and maintained by **Michał Zieliński**.

Feel free to reach out for questions, suggestions, or collaboration.
 
[GitHub: @Mayk-ITdS](https://github.com/Mayk-ITdS)  
Email: majk.develop@gmail.com  <!-- (replace with your real one if you want) -->
