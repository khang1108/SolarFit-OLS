"""
Part 1: OLS Implementation and Analysis

This package contains the core OLS implementation and related analysis functions
as required by the project specification.

Modules:
    ols_implementation - Core OLS solver and hat matrix
    model_evaluation - Model metrics and residual analysis
    inference - Coefficient inference and VIF
    regularization - Ridge regression
    cross_validation - k-fold cross-validation
    gauss_markov_sim - Gauss-Markov theorem verification
"""

__version__ = "1.0.0"

from .ols_implementation import ols_fit, hat_matrix, run_ols_analysis
from .model_evaluation import model_metrics, residual_plots
from .inference import coef_inference, vif
from .regularization import ridge_fit, ridge_trace
from .cross_validation import kfold_cv, kfold_cv_ridge, model_selection_cv
from .gauss_markov_sim import simulate_gauss_markov, verify_assumptions

__all__ = [
    'ols_fit',
    'hat_matrix',
    'run_ols_analysis',
    'model_metrics',
    'residual_plots',
    'coef_inference',
    'vif',
    'ridge_fit',
    'ridge_trace',
    'kfold_cv',
    'kfold_cv_ridge',
    'model_selection_cv',
    'simulate_gauss_markov',
    'verify_assumptions',
]
