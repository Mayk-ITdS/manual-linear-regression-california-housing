# Manual Linear Regression on California Housing Data

This project implements a full end-to-end pipeline for predicting house prices in California using a manually constructed linear regression model — without relying on machine learning libraries like `scikit-learn`.

The goal is to deeply understand each step of the regression process, including:
- Data extraction and normalization (ETL)
- Exploratory Data Analysis (EDA)
- Pearson correlation analysis
- Manual train/test splitting
- Linear regression modeling built from scratch (matrix algebra)

## Dataset

The dataset is sourced from the `fetch_california_housing()` function available in `scikit-learn.datasets`. It contains information about California housing districts, with features such as:

- Median income
- House age
- Average number of rooms
- Geographic coordinates (latitude & longitude)
- Population
- And more...

The target variable is the **median house value**, expressed in hundreds of thousands of dollars.

## Project Structure 

```manual-linear-regression-california-housing/
├── README.md # Project overview and documentation
├── regression.py # Core script with EDA and modeling
├── src/ # (optional) Reusable functions and modules
│ └── regression_manual.py # Planned: custom linear regression
├── plots/ # Visualizations: correlation, predictions, etc.
├── notebooks/ # (optional) Jupyter experiments
└── .gitignore # Exclude unnecessary files from Git
```


## Learning & Repository Roadmap

This repository follows a dual-track learning path:

- Classic ML — to understand interpretability and structure
- Neural Networks — to explore representation and depth

| ML Path                           | Neural Network Path                        |
|-----------------------------------|--------------------------------------------|
| Manual linear regression          |  Single neuron (1 layer, `Linear`)         |
|  L1/L2 regularization             | Activations: ReLU, Sigmoid               |
| PCA, RFE (feature selection)      | MLP (2–3 layers)                         |
| Non-linear regression (functions) | Dropout, BatchNorm                       |
| Tree-based models / ensembles   | ML vs NN performance comparison           |

Each stage builds on the previous, reinforcing core intuition through implementation.

## Status

Data loaded  

Z-score normalization

Manual Pearson correlation  

Manual implementation of Linear Regression (matrix algebra)

## Author
[Mayk-ITdS]  

Aspiring machine learning researcher — exploring core ML ideas by building everything from scratch.  
This repository is a part of my self-directed learning journey, with professor-style rigor.

