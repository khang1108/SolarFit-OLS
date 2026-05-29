"""
Monte Carlo Simulation to Demonstrate the Gauss-Markov Theorem.

This module simulates the Gauss-Markov theorem by repeatedly:
1. Fixing the design matrix X and true parameters β
2. Generating random noise ε ~ N(0, σ²I)
3. Creating observation vectors y = Xβ + ε
4. Computing OLS estimates and comparing with alternative estimators

The simulation verifies two key properties:
- Unbiasedness: E[β̂_OLS | X] = β
- Optimality: Var(β̂_OLS | X) ≤ Var(β̃_Alt | X) for any unbiased linear estimator β̃_Alt
"""

import numpy as np
from ols_implementation import ols_fit, hat_matrix, _matmul, _matvec


def make_perturbation(X, B):
    """
    Create a perturbation matrix A = B(I - H) with A X = 0.

    Since (I - H)X = 0, the resulting A satisfies the unbiasedness constraint.

    Parameters
    ----------
    X : list of list
        Design matrix as nested lists
    B : list of list
        Random perturbation matrix

    Returns
    -------
    list of list
        Perturbation matrix A = B(I - H)
    """
    H = hat_matrix(X).H
    n = len(X)
    ImH = [[(1.0 if i == j else 0.0) - H[i][j] for j in range(n)] for i in range(n)]
    return _matmul(B, ImH)


def alt_fit(X, y, A):
    """
    Compute alternative linear unbiased estimator: β̃_Alt = β̂_OLS + Ay.

    Parameters
    ----------
    X : list of list
        Design matrix
    y : list
        Observation vector
    A : list of list
        Perturbation matrix (must satisfy AX = 0)

    Returns
    -------
    list
        Coefficient vector β̃_Alt
    """
    beta_ols = ols_fit(X, y).beta_hat
    Ay = _matvec(A, y)
    return [b + a for b, a in zip(beta_ols, Ay)]


def run_monte_carlo(n=30, beta_true=(2.0, 1.0, -0.5), sigma=1.0,
                    n_rep=8000, seed=2024, alt_scale=0.06):
    """
    Run Monte Carlo simulation to approximate E[β̂] and Var(β̂).

    Parameters
    ----------
    n : int
        Number of observations
    beta_true : tuple
        True parameters (intercept, slope1, slope2)
    sigma : float
        Standard deviation of noise
    n_rep : int
        Number of Monte Carlo replications
    seed : int
        Random seed for reproducibility
    alt_scale : float
        Scaling factor for perturbation matrix B

    Returns
    -------
    dict
        Dictionary containing:
        - X: Fixed design matrix (numpy array)
        - A: Perturbation matrix (numpy array)
        - beta_true: True parameters
        - sigma: Noise standard deviation
        - n, k, n_rep: Dimensions and iterations
        - beta_ols: Array of shape (n_rep, k) with OLS estimates
        - beta_alt: Array of shape (n_rep, k) with alternative estimates
    """
    rng = np.random.default_rng(seed)
    beta_true = np.asarray(beta_true, float)
    k = len(beta_true)

    # Fixed design X (theorem is conditional on X)
    X_np = np.column_stack([np.ones(n), rng.normal(size=(n, k - 1))])
    X_list = X_np.tolist()

    # Fixed perturbation A = B(I - H) with AX = 0, built once from X
    B = (alt_scale * rng.normal(size=(k, n))).tolist()
    A = make_perturbation(X_list, B)
    A_np = np.asarray(A)

    # Storage for results
    beta_ols = np.empty((n_rep, k))
    beta_alt = np.empty((n_rep, k))

    # Fixed signal component
    mu = X_np @ beta_true

    # Monte Carlo loop
    for r in range(n_rep):
        eps = rng.normal(scale=sigma, size=n)
        y_vec = (mu + eps).tolist()
        beta_ols[r] = ols_fit(X_list, y_vec).beta_hat
        beta_alt[r] = alt_fit(X_list, y_vec, A)

    return {
        "X": X_np,
        "A": A_np,
        "beta_true": beta_true,
        "sigma": sigma,
        "n": n,
        "k": k,
        "n_rep": n_rep,
        "beta_ols": beta_ols,
        "beta_alt": beta_alt,
    }


def verify_with_numpy_sklearn(X_np, y_one):
    """
    Verify OLS implementation against NumPy and scikit-learn.

    Parameters
    ----------
    X_np : numpy.ndarray
        Design matrix
    y_one : list or array
        Observation vector

    Returns
    -------
    dict
        Dictionary with max errors from NumPy and scikit-learn
    """
    from sklearn.linear_model import LinearRegression

    beta_scratch = np.asarray(ols_fit(X_np.tolist(), y_one).beta_hat)
    beta_numpy, *_ = np.linalg.lstsq(X_np, y_one, rcond=None)
    beta_sklearn = LinearRegression(fit_intercept=False).fit(X_np, y_one).coef_

    return {
        "error_numpy": np.max(np.abs(beta_scratch - beta_numpy)),
        "error_sklearn": np.max(np.abs(beta_scratch - beta_sklearn)),
    }


if __name__ == "__main__":
    # Run the simulation with default parameters
    print("Running Monte Carlo simulation...")
    results = run_monte_carlo()

    # Compute summary statistics
    beta_ols_mean = results["beta_ols"].mean(axis=0)
    beta_alt_mean = results["beta_alt"].mean(axis=0)
    beta_ols_var = results["beta_ols"].var(axis=0, ddof=1)
    beta_alt_var = results["beta_alt"].var(axis=0, ddof=1)

    print("\nMean estimates (should match true values):")
    print(f"β_true:     {results['beta_true']}")
    print(f"β̂_OLS:      {beta_ols_mean}")
    print(f"β̃_Alt:      {beta_alt_mean}")

    print("\nVariance comparison (Alt should be larger):")
    print(f"Var(β̂_OLS): {beta_ols_var}")
    print(f"Var(β̃_Alt): {beta_alt_var}")
    print(f"Ratio:      {beta_alt_var / beta_ols_var}")
