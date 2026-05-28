"""
OLS Solvers -- Data Fitting and Ordinary Least Squares

Core functions (pure Python, no NumPy):
    ols_fit(X, y)       -- OLS coefficients and noise-variance estimate.
    hat_matrix(X, tol)  -- Projection matrix H and property checks.

NumPy is used only for verification and testing.
"""

import copy
from dataclasses import dataclass
from math import sqrt
from time import perf_counter
from typing import List, Optional, Tuple


EPS_PIVOT        = 1e-12   # pivot threshold for singularity detection
DEFAULT_TOL_IDEM = 1e-10   # Frobenius tolerance for idempotent check

@dataclass
class OLSResult:
    """Unified result container for ols_fit."""
    method: str               # "OLS-NormalEquations"
    beta_hat: List[float]     # shape (p+1,)
    sigma2_hat: float         # RSS / (n - p - 1)
    y_hat: List[float]        # shape (n,)
    residuals: List[float]    # shape (n,)
    rss: float                # ||e||^2
    dof: int                  # n - p - 1
    success: bool
    runtime_sec: float
    message: str


@dataclass
class HatMatrixResult:
    """Unified result container for hat_matrix."""
    method: str                  # "HatMatrix"
    H: List[List[float]]         # shape (n, n)
    sym_err: float               # ||H - H^T||_inf
    idem_err: float              # ||H^2 - H||_F
    rank_H: int                  # expected p+1
    trace_H: float               # expected p+1
    success: bool
    runtime_sec: float
    message: str


def _validate_inputs(
    X: List[List[float]], y: Optional[List[float]] = None
) -> Tuple[List[List[float]], Optional[List[float]]]:
    """Validate X and y; return float copies."""
    if not isinstance(X, list) or not X or not isinstance(X[0], list):
        raise ValueError("X must be a 2-D list of lists.")
    n, k = len(X), len(X[0])
    if any(len(row) != k for row in X):
        raise ValueError("All rows of X must have the same length.")
    if n < k:
        raise ValueError(f"n={n} must be >= k={k} (underdetermined system).")
    X_copy = [[float(v) for v in row] for row in X]
    if y is None:
        return X_copy, None
    if len(y) != n:
        raise ValueError(f"len(y)={len(y)} must equal n={n}.")
    return X_copy, [float(v) for v in y]


def _transpose(A: List[List[float]]) -> List[List[float]]:
    """Transpose matrix A."""
    m, n = len(A), len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]


def _matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Multiply A (m x k) by B (k x n), return C (m x n)."""
    m, k, n = len(A), len(A[0]), len(B[0])
    return [[sum(A[i][p] * B[p][j] for p in range(k)) for j in range(n)]
            for i in range(m)]


def _matvec(A: List[List[float]], x: List[float]) -> List[float]:
    """Multiply matrix A by vector x."""
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def _vecsub(a: List[float], b: List[float]) -> List[float]:
    """Subtract vector b from vector a."""
    return [a[i] - b[i] for i in range(len(a))]


def _dot(x: List[float], y: List[float]) -> float:
    """Compute dot product of vectors x and y."""
    return sum(xi * yi for xi, yi in zip(x, y))


def _mat_sub(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Subtract matrix B from matrix A."""
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _norm_fro(A: List[List[float]]) -> float:
    """Compute Frobenius norm of matrix A."""
    return sqrt(sum(A[i][j] ** 2 for i in range(len(A)) for j in range(len(A[0]))))


def _norm_inf_mat(A: List[List[float]]) -> float:
    """Compute infinity norm of matrix A."""
    return max(abs(A[i][j]) for i in range(len(A)) for j in range(len(A[0])))


def _trace(A: List[List[float]]) -> float:
    """Compute trace of square matrix A."""
    return sum(A[i][i] for i in range(len(A)))


def _mat_inv(A: List[List[float]]) -> List[List[float]]:
    """Invert A via Gauss-Jordan on the augmented matrix [A | I]."""
    n = len(A)
    aug = [A[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < EPS_PIVOT:
            raise ValueError(f"Singular matrix: zero pivot at column {col}.")
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot_val = aug[col][col]
        aug[col] = [v / pivot_val for v in aug[col]]
        for row in range(n):
            if row == col:
                continue
            f = aug[row][col]
            aug[row] = [aug[row][j] - f * aug[col][j] for j in range(2 * n)]
    return [row[n:] for row in aug]


def _mat_rank(A: List[List[float]], tol: float = EPS_PIVOT) -> int:
    """Compute rank of A via Gaussian elimination."""
    m, n = len(A), len(A[0])
    mat = copy.deepcopy(A)
    rank, used_row = 0, 0
    for col in range(n):
        pivot_row = next((r for r in range(used_row, m) if abs(mat[r][col]) > tol), None)
        if pivot_row is None:
            continue
        mat[used_row], mat[pivot_row] = mat[pivot_row], mat[used_row]
        for r in range(used_row + 1, m):
            if abs(mat[used_row][col]) < tol:
                break
            f = mat[r][col] / mat[used_row][col]
            mat[r] = [mat[r][j] - f * mat[used_row][j] for j in range(n)]
        rank += 1
        used_row += 1
    return rank


# Core solvers
def ols_fit(X: List[List[float]], y: List[float]) -> OLSResult:
    """
    Compute OLS estimate via Normal Equations.

    Model : y = X beta + eps,  eps ~ (0, sigma^2 I_n)

    Algorithm:
        1. G = X^T X,  q = X^T y
        2. Check rank(G) == k
        3. beta_hat = G^{-1} q
        4. y_hat = X beta_hat,  e = y - y_hat,  RSS = e.e
        5. sigma2_hat = RSS / (n - p - 1)

    Parameters:
        X : List[List[float]], shape (n, p+1)  -- design matrix (first col = 1).
        y : List[float], shape (n,)            -- response vector.

    Returns:
        OLSResult
    """
    start = perf_counter()
    X_mat, y_vec = _validate_inputs(X, y)
    n, k = len(X_mat), len(X_mat[0])
    p = k - 1

    try:
        # Step 1: G = X^T X,  q = X^T y
        Xt = _transpose(X_mat)
        G  = _matmul(Xt, X_mat)
        q  = _matvec(Xt, y_vec)

        # Step 2: Check invertibility
        if _mat_rank(G) < k:
            return OLSResult(
                method="OLS-NormalEquations", beta_hat=[], sigma2_hat=float("nan"),
                y_hat=[], residuals=[], rss=float("nan"), dof=n - p - 1,
                success=False, runtime_sec=perf_counter() - start,
                message="X^T X is not invertible (multicollinearity).",
            )

        # Step 3: beta_hat = (X^T X)^{-1} X^T y
        beta_hat = _matvec(_mat_inv(G), q)

        # Step 4: Residuals and RSS
        y_hat     = _matvec(X_mat, beta_hat)
        residuals = _vecsub(y_vec, y_hat)
        rss       = _dot(residuals, residuals)

        # Step 5: sigma^2 = RSS / (n - p - 1)
        dof        = n - p - 1
        sigma2_hat = rss / dof

        return OLSResult(
            method="OLS-NormalEquations", beta_hat=beta_hat, sigma2_hat=sigma2_hat,
            y_hat=y_hat, residuals=residuals, rss=rss, dof=dof, success=True,
            runtime_sec=perf_counter() - start,
            message=f"Solved. RSS={rss:.6g}, sigma2={sigma2_hat:.6g}, dof={dof}.",
        )

    except Exception as exc:
        return OLSResult(
            method="OLS-NormalEquations", beta_hat=[], sigma2_hat=float("nan"),
            y_hat=[], residuals=[], rss=float("nan"), dof=n - p - 1,
            success=False, runtime_sec=perf_counter() - start,
            message=f"ols_fit failed: {exc}",
        )


def hat_matrix(X: List[List[float]], tol: float = DEFAULT_TOL_IDEM) -> HatMatrixResult:
    """
    Compute the hat matrix H = X (X^T X)^{-1} X^T and verify properties.

    H is the orthogonal projector onto C(X):
        y_hat = H y,   e = (I - H) y

    Properties checked:
        (i)  Idempotent : ||H^2 - H||_F <= tol
        (ii) Symmetric  : ||H - H^T||_inf  (reported)
        (iii) rank(H) = p+1,  tr(H) = p+1

    Algorithm:
        1. G_inv = (X^T X)^{-1}
        2. H = X G_inv X^T
        3. sym_err, idem_err, rank_H, trace_H

    Parameters:
        X   : List[List[float]], shape (n, p+1)
        tol : float  -- idempotent tolerance (default 1e-10)

    Returns:
        HatMatrixResult
    """
    start = perf_counter()
    X_mat, _ = _validate_inputs(X)
    n, k = len(X_mat), len(X_mat[0])

    try:
        # Step 1: (X^T X)^{-1}
        Xt    = _transpose(X_mat)
        G_inv = _mat_inv(_matmul(Xt, X_mat))

        # Step 2: H = X (X^T X)^{-1} X^T
        H = _matmul(_matmul(X_mat, G_inv), Xt)

        # Step 3: Property checks
        sym_err  = _norm_inf_mat(_mat_sub(H, _transpose(H)))
        idem_err = _norm_fro(_mat_sub(_matmul(H, H), H))
        rank_H   = _mat_rank(H)
        trace_H  = _trace(H)

        ok = idem_err <= tol
        return HatMatrixResult(
            method="HatMatrix", H=H, sym_err=sym_err, idem_err=idem_err,
            rank_H=rank_H, trace_H=trace_H, success=ok,
            runtime_sec=perf_counter() - start,
            message=(
                f"{'[OK]' if ok else '[WARN]'} ||H^2-H||_F={idem_err:.2e}, "
                f"||H-H^T||_inf={sym_err:.2e}, "
                f"rank={rank_H} (exp {k}), tr={trace_H:.4f} (exp {k})."
            ),
        )

    except Exception as exc:
        return HatMatrixResult(
            method="HatMatrix", H=[[]], sym_err=float("nan"), idem_err=float("nan"),
            rank_H=-1, trace_H=float("nan"), success=False,
            runtime_sec=perf_counter() - start,
            message=f"hat_matrix failed: {exc}",
        )


def run_ols_analysis(
    X: List[List[float]], y: List[float], tol_idem: float = DEFAULT_TOL_IDEM
) -> dict:
    """Run ols_fit + hat_matrix and return results."""
    return {"ols": ols_fit(X, y), "hat": hat_matrix(X, tol=tol_idem)}


if __name__ == "__main__":
    import sys, dataclasses
    import numpy as np

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    np.random.seed(42)
    n_obs, p_feat = 20, 2
    beta_true     = np.array([2.0, 1.0, -0.5])
    X_raw         = np.random.randn(n_obs, p_feat)
    X_np          = np.column_stack([np.ones(n_obs), X_raw])
    y_np          = X_np @ beta_true + 0.5 * np.random.randn(n_obs)

    X_list, y_list = X_np.tolist(), y_np.tolist()
    results = run_ols_analysis(X_list, y_list)

    for name, res in results.items():
        print(f"\n{'='*55}\n  {name.upper()}\n{'='*55}")
        for field, val in dataclasses.asdict(res).items():
            if isinstance(val, list):
                if val and isinstance(val[0], list):
                    print(f"  {field}: <{len(val)}x{len(val[0])} matrix>")
                elif len(val) <= 6:
                    print(f"  {field}: {[round(v, 6) for v in val]}")
                else:
                    print(f"  {field}: {[round(v, 6) for v in val[:4]]} ... (n={len(val)})")
            else:
                print(f"  {field}: {val}")

    # --- Verification against NumPy ---
    ols_res = results["ols"]
    hat_res = results["hat"]
    print(f"\n{'='*55}\n  Verification\n{'='*55}")

    beta_np, _, _, _ = np.linalg.lstsq(X_np, y_np, rcond=None)
    diff_beta = float(np.max(np.abs(np.array(ols_res.beta_hat) - beta_np)))
    print(f"  ols_fit   ||beta_hat - numpy||_inf  = {diff_beta:.2e}  {'PASSED' if diff_beta < 1e-8 else 'FAILED'}")

    H_np   = X_np @ np.linalg.inv(X_np.T @ X_np) @ X_np.T
    diff_H = float(np.max(np.abs(np.array(hat_res.H) - H_np)))
    print(f"  hat_matrix ||H_ours - H_numpy||_inf = {diff_H:.2e}  {'PASSED' if diff_H < 1e-8 else 'FAILED'}")

    print(f"\n  beta_true : {beta_true.tolist()}")
    print(f"  beta_hat  : {[round(v, 6) for v in ols_res.beta_hat]}")
