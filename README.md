
# Custom Linear Regression Engine - Built Entirely From Scratch

# Overview

This project implements linear regression models for predicting median housing values based on the California Housing dataset.
Unlike typical scikit-learn implementations, this project builds regression from scratch, covering:

- manual Ordinary Least Squares (OLS),

- Gradient Descent optimization,

- Ridge Regression (L2 regularization),

- fully custom ETL + Feature Engineering + Pipeline.

The project serves as both a learning exercise (understanding ML fundamentals deeply) and a solid codebase for extending toward more advanced models (e.g., Lasso, Logistic Regression, Neural Nets).

#### Best result: R² ≈ 0.692 (OLS, Gradient Descent)


## Pipeline Workflow

The pipeline (pipeline.py) orchestrates the following steps:

1. **ETL / Feature Engineering**
   - Assigns **geographical categories** (`Category`) from latitude/longitude using bounding boxes (`bounds.py`).
   - One-hot encodes categories (`OHE`), dropping one dummy column to avoid multicollinearity.
   - Filters large/small sample categories and computes descriptive statistics.

2. **Transformations**
   - Applies **Z-score scaling** to numerical features (`MedInc`, `HouseAge`, `AveRooms`, etc.).
   - Computes **Pearson** and **Spearman** correlations (features vs. target, and feature-to-feature).

3. **Data Storage**
   - Saves intermediate datasets for transparency and reproducibility:
     - `df_updated_features.csv` → raw after ETL  
     - `df_numeric_scaled.csv` → numerical features after Z-score  
     - `df_final.csv` → final dataset (numeric scaled + OHE)  

4. **Modeling**
   - Fits regression models (OLS, Ridge) on final dataset.
   - Exports model parameters (weights, bias, metrics) to JSON.


## Models Implemented

###  Linear Regression (OLS)
- Normal equations implementation.
- Gradient descent with early stopping.
- Computes MSE, MAE, R².

###  Ridge Regression (L2)
- Closed-form solution:
β_ridge = (XᵀX + λI)⁻¹ Xᵀy

- Implemented with manual Gauss–Jordan inversion.
- Gradient descent with L2 penalty.

###  Lasso (planned)
- To be implemented via **coordinate descent**.



## Quick Start
### 1. Install dependencies:
```bash

pip install -r requirements.txt
```
### 2. Run pipeline
```bash

python pipeline.py
```

### 3. Run the training interface:

```bash

python src/run_regressor.py
```

### You`ll see a menu:
```
        =================================================
             Welcome to Majk's Linear Regression Lab
              (OLS & Ridge — Normal Eq. / Gradient)
        =================================================
        
=== Available Solvers ===
  1. OLS (Normal Equation)
  2. Ridge (Normal Equation)
  3. OLS (Gradient Descent)
  4. Ridge (Gradient Descent)

Choose solver (1-4): 
```
### After training, the program saves:

Best model → /data/models/model_{solver}_{timestamp}.csv

Training history → /data/models/history_{solver}_{timestamp}.csv

Training log → /data/logs/training_{solver}_{timestamp}.log 

## Outputs
All artifacts are versioned with timestamps for reproducibility.
- All outputs stored in /data/ subfolders.

- Processed data → /data/processed/

- df_numeric_scaled.csv (all features)

- df_zscore_final.csv (selected regression-ready features)

- Correlation matrices (feats_corr_matrix.csv, feats_target_corr.csv)

- Models → /data/models/

- model_{solver}_{timestamp}.csv — weights, bias, metrics

- history_{solver}_{timestamp}.csv — per-epoch training history

- Logs → /data/logs/

- training_{solver}_{timestamp}.log — metadata + R²


```
Files created:
- model_OLS_GD_24-08-2025_19-20-56.csv — Best model
- history_OLS_GD_24-08-2025_19-20-56.csv — Training history
- training_OLS_GD_24-08-2025_19-20-56.log — Training log
```
### Example Results

```
Example metrics (OLS on z-scored data):

| Metric                 | Value     | Description                                |
| ---------------------- | --------- | ------------------------------------------ |
| Best MSE (z-score)     | ~0.3525   | Average squared error on standardized data |
| RMSE                   | ~0.59     | ≈ $59,000 prediction error                 |
| Absolute Error         | ~1.5–2.0  | Mostly within $150k–200k from truth        |
| Epochs                 | ~60–120   | Training converges early                   |
| Stability              | Moderate  | Low variance in training, some outliers    |

```


### 3. Generate visualizations

```
from visualizer import Visualizer

viz = Visualizer(X, Y, X_zscore, Y_zscore, df_xy_corr, feature_names, df_result)
viz.correlation_plot()
viz.prediction_vs_target()
viz.show_distribution_by_category()

```

##  Project Structure

```
src/
│
├── ETL/
│   ├── extract.py        # Load dataset
│   ├── transform.py      # Feature engineering, normalization, correlations
│   └── bounds.py         # Filtering & cleaning rules
│
├── utils/
│   └── factory.py        # Model/solver factory
│
├── pipeline.py           # Full preprocessing pipeline
├── run_regressor.py      # Main entrypoint (training interface)
├── linear_regression.py  # Core model + solvers
│
data/
├── processed/            # Processed datasets (zscored, correlation matrices)
├── models/               # Saved models & histories
└── summary/              # Data summaries

```


## ML Roadmap

| Classical ML Path                | Neural Network Path (extension)               |
|----------------------------------|-----------------------------------------------|
| Manual linear regression         | Single-layer perceptron (manual implementation) |
| Pearson & Spearman correlation   | Nonlinear activation functions (ReLU, Sigmoid) |
| Gradient Descent optimization    | Multi-Layer Perceptron (MLP)                  |
| Regularization (L1 / L2)         | Dropout, BatchNorm                            |
| PCA, RFE (feature selection)     | Learned feature representations / embeddings   |
| Model benchmarking (OLS, Ridge)  | ML vs NN performance comparison               |


## Status
| Component                               | Status |
|-----------------------------------------|--------|
| Data loading & normalization (ETL)      | Complete |
| Pearson correlation (features/target)   | Complete |
| Spearman correlation                    | Complete |
| Linear regression (Normal Equations)    | Complete |
| Linear regression (Gradient Descent)    | Complete |
| Feature engineering                     | Complete |
| Custom training loop & loss             | Complete |
| Evaluation metrics (R², MSE, RMSE)      | Complete |
| Weight saving/loading                   | Complete |
| Single-layer neural network             | In Progress |
| Test set evaluation                     | To Do  |

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


-  Matrix algebra:
```text 
(β = (XᵀX)^-1 Xᵀy)
  ```

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

- **Geographical categories**: created from lat/lon using bounding boxes.  
- **One-hot encoding** with drop-one strategy (avoiding dummy variable trap).  
- **Scaling**: Z-score applied to numeric features before correlation and modeling.  
- **Quartiles of MedHouseVal**: generated for descriptive analysis (not used as model feature).

Custom feature transformations implemented:

- RoomPerPerson = (AveRooms - AveBedrooms) / AveOccup → living space per person

- EstmIncPerson = MedInc / AveOccup → estimated household income per person

- PopulationPerHousehold = Population / AveOccup → average household size

- MedInc_sq = MedInc² → non-linear effect of income

- EstmIncPerson_sq = EstmIncPerson² → non-linear effect of per-person income

Additional transformations:

- One-hot encoding for categorical region labels (e.g. SoCal, LA Metro, Bay Area, Central Valley, …)

- Interaction terms between income and region (e.g. MedInc_x_cat_LA Metro)

All numeric features are standardized to zero mean and unit variance before training.
## Features and Techniques

- Linear Regression via manual gradient descent

- Feature engineering: new domain-specific fields added

- Spearman and Pearson correlation calculations

- Z-score standardization + Spearman rank normalization

- Training loop with early stopping

- Model persistence in JSON (manual loading + inference)

- Modular visualization layer for predictions and errors



### Next Steps

- Integrate sklearn’s regression for baseline comparison
- Implement QR / SVD solvers for stability
- Benchmark vs. Random Forests / XGBoost
- Prepare Jupyter notebooks with derivations
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
