"""Các hàm phân phối xác suất cần cho suy luận hồi quy, triển khai từ đầu.

Module này chỉ dùng Python standard library. SciPy không được dùng trong luồng
tính toán chính; nó chỉ đóng vai trò oracle trong ``verify_pvalues.py`` và test.
"""

from math import exp, isfinite, lgamma, log, log1p


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Tính hàm beta không đầy đủ chuẩn hóa I_x(a, b)."""
    if a <= 0.0 or b <= 0.0:
        raise ValueError("a and b must be positive")
    if not 0.0 <= x <= 1.0:
        raise ValueError("x must be between 0 and 1")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    log_factor = (
        lgamma(a + b)
        - lgamma(a)
        - lgamma(b)
        + a * log(x)
        + b * log1p(-x)
    )
    factor = exp(log_factor)
    if x < (a + 1.0) / (a + b + 2.0):
        result = factor * _beta_continued_fraction(a, b, x) / a
    else:
        result = 1.0 - factor * _beta_continued_fraction(b, a, 1.0 - x) / b
    return min(1.0, max(0.0, result))


def student_t_two_sided_pvalue(t_stat: float, dof: int) -> float:
    """Tính p-value hai phía P(|T_dof| >= |t_stat|) từ đầu."""
    if dof <= 0:
        raise ValueError("dof must be positive")
    if t_stat != t_stat:
        return float("nan")
    if not isfinite(t_stat):
        return 0.0

    x = dof / (dof + abs(t_stat) ** 2)
    return regularized_incomplete_beta(x, dof / 2.0, 0.5)


def student_t_critical(alpha_half: float, dof: int) -> float:
    """Tính phân vị t dương có xác suất đuôi trên bằng ``alpha_half``."""
    if not 0.0 < alpha_half < 0.5:
        raise ValueError("alpha_half must be between 0 and 0.5")
    if dof <= 0:
        raise ValueError("dof must be positive")

    target_pvalue = 2.0 * alpha_half
    lower = 0.0
    upper = 1.0
    while student_t_two_sided_pvalue(upper, dof) > target_pvalue:
        upper *= 2.0

    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if student_t_two_sided_pvalue(midpoint, dof) > target_pvalue:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def f_survival_probability(
    f_stat: float,
    df_numerator: int,
    df_denominator: int,
) -> float:
    """Tính p-value đuôi phải P(F >= f_stat) của kiểm định F."""
    if df_numerator <= 0 or df_denominator <= 0:
        raise ValueError("F-distribution degrees of freedom must be positive")
    if f_stat != f_stat:
        return float("nan")
    if f_stat < 0.0:
        raise ValueError("f_stat must be non-negative")
    if not isfinite(f_stat):
        return 0.0
    if f_stat == 0.0:
        return 1.0

    x = df_denominator / (df_denominator + df_numerator * f_stat)
    return regularized_incomplete_beta(
        x,
        df_denominator / 2.0,
        df_numerator / 2.0,
    )


def _beta_continued_fraction(
    a: float,
    b: float,
    x: float,
    max_iterations: int = 200,
    tolerance: float = 3e-14,
) -> float:
    """Tính continued fraction của I_x(a,b) bằng thuật toán Lentz."""
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    fraction = d

    for iteration in range(1, max_iterations + 1):
        doubled = 2 * iteration
        coefficient = (
            iteration * (b - iteration) * x
            / ((qam + doubled) * (a + doubled))
        )
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        fraction *= d * c

        coefficient = -(
            (a + iteration) * (qab + iteration) * x
            / ((a + doubled) * (qap + doubled))
        )
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        fraction *= delta
        if abs(delta - 1.0) <= tolerance:
            return fraction

    raise ArithmeticError("incomplete beta continued fraction did not converge")
