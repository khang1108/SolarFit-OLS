<div align="center">

# ☀️ SolarFit-OLS

### Data Fitting and Ordinary Least Squares for Solar Radiation Prediction

*From first-principles regression theory to real-world weather-station machine learning.*

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/Numerical-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Data-Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Validation-scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/ML-LightGBM-6DB33F?style=flat-square)
![CatBoost](https://img.shields.io/badge/ML-CatBoost-FFCC00?style=flat-square)
![Course](https://img.shields.io/badge/HCMUS-MTH00051-005BBB?style=flat-square)

[Overview](#overview) · [Dataset](#dataset) · [Methodology](#methodology) · [Models](#models) · [Project Structure](#project-structure) · [Getting Started](#getting-started)

</div>

---

## Overview

**SolarFit-OLS** is a course project for **Data Fitting và Phương Pháp OLS** at HCMUS.

The project studies incoming solar radiation prediction using real TAHMO weather-station observations. It begins with the mathematical foundation of **Ordinary Least Squares**, then builds a complete applied regression pipeline and compares interpretable statistical models with modern machine-learning baselines.

The central idea is simple:

> Use OLS to understand data fitting deeply, then use modern ML to see how far practical prediction can be pushed.

---

## Why This Project Matters

Incoming solar radiation is important for renewable energy, agriculture, weather monitoring, and climate-aware planning. In real systems, however, sensors drift, time-series observations are irregular, and prediction quality varies across stations, months, and geographic regions.

This makes the TAHMO problem a strong real-world case study for data fitting:

- the target is continuous;
- the data come from real weather stations;
- observations are temporal and geographically distributed;
- the task naturally exposes missing-target periods, outliers, seasonal effects, and nonlinear behavior;
- linear models are interpretable but limited;
- gradient-boosting models are powerful but less transparent.

---

## Dataset

This repository uses data from the **TAHMO Incoming Solar Radiation Prediction Challenge** on Zindi.

```text
data/
├── Train.csv
├── Test.csv
├── SampleSubmission.csv
├── dataset_data_dictionary.csv
└── tahmo_starter_notebook.ipynb
```

Key columns:

| Column | Meaning |
|---|---|
| `timestamp` | 15-minute observation timestamp |
| `radiation (W/m2)` | target variable in training data |
| `precipitation (mm)` | precipitation level |
| `relativehumidity (-)` | relative humidity |
| `temperature (degrees Celsius)` | air temperature |
| `station`, `station_name` | station identifiers |
| `country` | station country |
| `installation_height`, `elevation` | station metadata |
| `latitude`, `longitude` | geographic position |

The original competition asks participants to predict solar radiation values for withheld station-month observations.

---

## Methodology

The project is divided into two parts that mirror the course requirements.

### Part 1 — OLS From First Principles

We implement the core regression machinery directly from mathematical formulas.

| Component | Purpose |
|---|---|
| `ols_fit(X, y)` | estimate coefficients using the normal equation |
| `hat_matrix(X)` | compute the projection matrix and verify idempotence |
| `model_metrics(y, y_hat, p)` | compute RSS, TSS, R², adjusted R², F-statistic |
| `coef_inference(...)` | compute standard errors, t-statistics, p-values, confidence intervals |
| `vif(X)` | diagnose multicollinearity |
| `ridge_fit(X, y, lambda)` | implement Ridge regression from scratch |
| `kfold_cv(X, y, k)` | perform manual cross-validation |
| residual plots | diagnose linearity, normality, variance, and influence |
| Monte Carlo simulation | illustrate the Gauss-Markov theorem |

### Part 2 — Real-World Solar Radiation Modeling

We apply the data-fitting workflow to the TAHMO dataset.

```mermaid
flowchart LR
    A[Raw TAHMO Data] --> B[EDA]
    B --> C[Feature Engineering]
    C --> D[Preprocessing Pipeline]
    D --> E[OLS / Ridge / Lasso]
    D --> F[Modern ML Baselines]
    E --> G[Evaluation]
    F --> G
    G --> H[Residual Analysis and Report]
```

Planned applied workflow:

1. Exploratory data analysis
2. Missing-value and temporal-gap discussion
3. Time, station, and geographic feature engineering
4. Train/test split with time-series awareness
5. Scaling and encoding
6. OLS baseline
7. OLS with variable selection using p-values or VIF
8. Ridge/Lasso with cross-validation
9. Polynomial or interaction regression
10. LightGBM/CatBoost benchmark
11. Residual diagnostics and interpretation
12. Optional Zindi submission generation

---

## Models

| Model | Role | Strength | Limitation |
|---|---|---|---|
| Basic OLS | required baseline | transparent and mathematically grounded | weak for nonlinear patterns |
| Selected OLS | interpretable refinement | removes unstable variables | still linear |
| Ridge / Lasso | regularized linear model | handles multicollinearity better | requires tuning λ |
| Polynomial Regression | nonlinear extension | captures interactions | can overfit |
| Random Forest | practical ML baseline | robust nonlinear learner | less statistically interpretable |
| LightGBM / CatBoost | modern benchmark | strong predictive performance | black-box tendency |

Evaluation metrics:

- MAE
- RMSE
- R²
- residual diagnostics
- coefficient or feature-importance analysis
- robustness across station/month splits

---

## Academic Contribution

This project is not only a leaderboard experiment. The main academic contribution is a complete, explainable data-fitting study:

- derive OLS from the least-squares objective;
- verify the normal equation implementation;
- analyze Gauss-Markov assumptions;
- estimate coefficient uncertainty;
- diagnose multicollinearity using VIF;
- compare ordinary and regularized regression;
- inspect residual patterns;
- explain why modern ML improves or fails relative to linear models.

---

## Project Structure

Target submission structure:

```text
Group_<ID>/
├── README.md
├── requirements.txt
├── report/
│   ├── report.pdf
│   └── report.tex
├── part1/
│   ├── ols_implementation.py
│   ├── ridge_lasso.py
│   ├── residual_analysis.py
│   ├── cross_validation.py
│   └── part1_notebook.ipynb
└── part2/
    ├── data/
    ├── data_pipeline.py
    ├── model_comparison.py
    ├── advanced_methods.py
    └── part2_notebook.ipynb
```

Current repository state:

```text
.
├── data/
│   ├── Train.csv
│   ├── Test.csv
│   ├── SampleSubmission.csv
│   ├── dataset_data_dictionary.csv
│   └── tahmo_starter_notebook.ipynb
├── req.txt
├── README.md
└── Toan UDTK_Project_2-Data Fitting va OLS.pdf
```

---

## Getting Started

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies after `requirements.txt` is finalized:

```bash
pip install -r requirements.txt
```

Suggested core packages:

```text
numpy
scipy
pandas
matplotlib
seaborn
scikit-learn
lightgbm
catboost
jupyter
```

---

## Expected Final Deliverables

- `report.pdf` with theory, experiments, results, and discussion
- Part 1 notebook demonstrating OLS theory and implementation
- Part 2 notebook demonstrating real-data modeling
- clean Python source files for reusable implementations
- model comparison table
- residual diagnostic figures
- reproducible preprocessing and training pipeline
- optional Zindi submission file

---

## Course Context

| Item | Detail |
|---|---|
| University | University of Science, VNU-HCM |
| Course | Toán Ứng Dụng và Thống Kê |
| Course Code | MTH00051 |
| Project | Đồ án 2 — Data Fitting và Phương Pháp OLS |
| Dataset | TAHMO Solar Radiation Prediction |

---

<div align="center">

### Built to connect mathematical understanding with real-world machine learning.

</div>
