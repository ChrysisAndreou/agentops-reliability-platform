"""
Statistical tests for A/B evaluation of AI agents.

Pure-statistical module providing hypothesis tests, Bayesian comparison,
sample size estimation, and confidence intervals. No external dependencies
beyond Python stdlib + numpy (for numerical stability). All tests are
two-sided where applicable and report both test statistics and p-values.

Tests implemented:
    - Chi-squared test of independence (2x2 contingency)
    - Fisher's exact test (2x2 contingency, exact — no chi-squared approximation)
    - Bayesian A/B test (beta-binomial conjugate model, probability of superiority)
    - Welch's t-test (unequal variance, continuous metrics)
    - Mann-Whitney U test (non-parametric, continuous/ordinal metrics)
    - Sample size estimation (power analysis for binomial proportions)
    - Confidence interval (Wilson score for binomial proportions)

References:
    - Kohavi, R., et al. "Online Controlled Experiments at Large Scale" (KDD 2013)
    - Deng, A., et al. "Trustworthy Online Controlled Experiments" (Cambridge, 2020)
    - VWO, C. "Bayesian A/B Testing: A Hypothesis Test that Uses Bayes' Theorem"
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Chi-squared test
# ---------------------------------------------------------------------------

def chi_squared_test(
    observed_a: list[int],
    observed_b: list[int],
    yates_correction: bool = True,
) -> Tuple[float, bool]:
    """Chi-squared test of independence for a 2x2 contingency table.

    Args:
        observed_a: [successes_a, failures_a]
        observed_b: [successes_b, failures_b]
        yates_correction: Apply Yates' continuity correction (default True).

    Returns:
        (p_value, significant) where significant is p_value < 0.05.
    """
    a_s, a_f = observed_a
    b_s, b_f = observed_b
    total = a_s + a_f + b_s + b_f
    if total == 0:
        return 1.0, False

    # Expected frequencies
    row1 = a_s + a_f
    row2 = b_s + b_f
    col1 = a_s + b_s
    col2 = a_f + b_f

    e11 = row1 * col1 / total
    e12 = row1 * col2 / total
    e21 = row2 * col1 / total
    e22 = row2 * col2 / total

    # Yates correction: subtract 0.5 from each |O - E|
    if yates_correction:
        chi2 = 0.0
        for o, e in [(a_s, e11), (a_f, e12), (b_s, e21), (b_f, e22)]:
            if o == 0 and e == 0:
                continue  # No contribution from empty cells
            chi2 += (abs(o - e) - 0.5) ** 2 / max(e, 1e-10)
    else:
        chi2 = 0.0
        for o, e in [(a_s, e11), (a_f, e12), (b_s, e21), (b_f, e22)]:
            if o == 0 and e == 0:
                continue
            chi2 += (o - e) ** 2 / max(e, 1e-10)

    p_value = _chi2_sf(chi2, df=1)
    return p_value, p_value < 0.05


# ---------------------------------------------------------------------------
# Fisher's exact test
# ---------------------------------------------------------------------------

def fisher_exact_test(
    a_s: int, a_f: int, b_s: int, b_f: int
) -> Tuple[float, float, bool]:
    """Fisher's exact test for a 2x2 contingency table.

    Computes the exact hypergeometric probability. Preferred over chi-squared
    when any cell count < 5 or total sample < 1000.

    Args:
        a_s: Successes for variant A.
        a_f: Failures for variant A.
        b_s: Successes for variant B.
        b_f: Failures for variant B.

    Returns:
        (p_value, odds_ratio, significant)
    """
    n = a_s + a_f + b_s + b_f
    if n == 0:
        return 1.0, 1.0, False

    # Odds ratio
    odds_ratio = (a_s * b_f) / max(a_f * b_s, 1e-10)
    if odds_ratio < 1e-10:
        odds_ratio = 0.0

    # Hypergeometric probability of observed table
    import math as _math

    def _log_comb(n_val, k_val):
        if k_val < 0 or k_val > n_val:
            return float("-inf")
        return _math.lgamma(n_val + 1) - _math.lgamma(k_val + 1) - _math.lgamma(n_val - k_val + 1)

    log_p_obs = (
        _log_comb(a_s + a_f, a_s)
        + _log_comb(b_s + b_f, b_s)
        - _log_comb(n, a_s + b_s)
    )

    # Sum probabilities of tables as extreme or more extreme
    # Two-sided: sum all tables with probability <= observed
    p_value = 0.0
    row1 = a_s + a_f
    col1 = a_s + b_s

    lo = max(0, col1 - b_s - b_f)
    hi = min(row1, col1)

    for x in range(lo, hi + 1):
        log_p_x = (
            _log_comb(row1, x)
            + _log_comb(n - row1, col1 - x)
            - _log_comb(n, col1)
        )
        p_x = math.exp(log_p_x)
        if p_x <= math.exp(log_p_obs) + 1e-14:
            p_value += p_x

    p_value = min(p_value, 1.0)
    return p_value, odds_ratio, p_value < 0.05


# ---------------------------------------------------------------------------
# Bayesian A/B test
# ---------------------------------------------------------------------------

def bayesian_ab_test(
    successes_a: int,
    trials_a: int,
    successes_b: int,
    trials_b: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    n_samples: int = 100_000,
    seed: Optional[int] = None,
) -> float:
    """Bayesian A/B test using the beta-binomial conjugate model.

    Computes P(B > A) — the probability that variant B has a higher
    success rate than variant A. Uses Monte Carlo sampling from the
    posterior beta distributions. Prior defaults to Beta(1,1) (uniform).

    Args:
        successes_a: Number of successes for variant A.
        trials_a: Total trials for variant A.
        successes_b: Number of successes for variant B.
        trials_b: Total trials for variant B.
        prior_alpha: Alpha parameter of the Beta prior (default 1.0).
        prior_beta: Beta parameter of the Beta prior (default 1.0).
        n_samples: Number of Monte Carlo samples (default 100,000).
        seed: Random seed for reproducibility.

    Returns:
        Probability that B > A, in [0, 1].
    """
    if trials_a == 0 or trials_b == 0:
        return 0.5

    failures_a = trials_a - successes_a
    failures_b = trials_b - successes_b

    rng = np.random.RandomState(seed)
    samples_a = rng.beta(
        prior_alpha + successes_a,
        prior_beta + failures_a,
        size=n_samples,
    )
    samples_b = rng.beta(
        prior_alpha + successes_b,
        prior_beta + failures_b,
        size=n_samples,
    )

    prob_b_better = float(np.mean(samples_b > samples_a))
    return prob_b_better


# ---------------------------------------------------------------------------
# Welch's t-test (unequal variance)
# ---------------------------------------------------------------------------

def welch_t_test(
    sample_a: list[float],
    sample_b: list[float],
) -> Tuple[float, float, bool]:
    """Welch's t-test for comparing means of two samples with unequal variance.

    Appropriate for continuous metrics (latency, cost, scores) where the
    two variants may have different variances. Does NOT assume equal variance.

    Args:
        sample_a: Numeric values for variant A.
        sample_b: Numeric values for variant B.

    Returns:
        (t_statistic, p_value, significant)
    """
    if len(sample_a) < 2 or len(sample_b) < 2:
        return 0.0, 1.0, False

    a = np.array(sample_a, dtype=np.float64)
    b = np.array(sample_b, dtype=np.float64)

    mean_a = np.mean(a)
    mean_b = np.mean(b)
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    n_a = len(a)
    n_b = len(b)

    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se < 1e-15:
        return 0.0, 1.0, False

    t_stat = (mean_a - mean_b) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = max(num / denom, 1.0)

    # Two-sided p-value from t-distribution
    p_value = 2.0 * _t_sf(abs(t_stat), df)
    return t_stat, p_value, p_value < 0.05


# ---------------------------------------------------------------------------
# Mann-Whitney U test
# ---------------------------------------------------------------------------

def mann_whitney_u(
    sample_a: list[float],
    sample_b: list[float],
) -> Tuple[float, float, bool]:
    """Mann-Whitney U test — non-parametric comparison of two independent samples.

    Does NOT assume normality. Ranks all values and compares rank sums.
    Appropriate for ordinal data or when normality is violated.

    Args:
        sample_a: Values for variant A.
        sample_b: Values for variant B.

    Returns:
        (u_statistic, p_value, significant)
    """
    n_a = len(sample_a)
    n_b = len(sample_b)
    if n_a == 0 or n_b == 0:
        return 0.0, 1.0, False

    # Combine, rank, compute U
    combined = [(v, 0) for v in sample_a] + [(v, 1) for v in sample_b]
    combined.sort(key=lambda x: x[0])

    ranks = []
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # +1 for 1-indexed ranks
        for _ in range(j - i):
            ranks.append((avg_rank, combined[i][1]))
            i += 1

    r1 = sum(r for r, grp in ranks if grp == 0)
    u1 = r1 - n_a * (n_a + 1) / 2.0
    u2 = n_a * n_b - u1
    u_stat = min(u1, u2)

    # Normal approximation
    mu = n_a * n_b / 2.0
    sigma = math.sqrt(n_a * n_b * (n_a + n_b + 1) / 12.0)

    if sigma < 1e-15:
        return u_stat, 0.5, False

    z = (u_stat - mu) / sigma
    # Two-sided
    p_value = 2.0 * _norm_sf(abs(z))
    return u_stat, p_value, p_value < 0.05


# ---------------------------------------------------------------------------
# Sample size estimation
# ---------------------------------------------------------------------------

def compute_sample_size(
    baseline_rate: float,
    min_detectable_effect: float,
    power: float = 0.80,
    alpha: float = 0.05,
) -> int:
    """Compute required sample size per variant for a binomial proportion A/B test.

    Uses the normal approximation for two-sample proportion test.

    Args:
        baseline_rate: Expected success rate of the control variant (0-1).
        min_detectable_effect: Smallest absolute difference to detect (0-1).
        power: Desired statistical power (default 0.80).
        alpha: Significance level (default 0.05).

    Returns:
        Required sample size per variant.

    Raises:
        ValueError: If baseline_rate or min_detectable_effect are out of range.
    """
    if not 0 < baseline_rate < 1:
        raise ValueError(f"baseline_rate must be in (0, 1), got {baseline_rate}")
    if not 0 < min_detectable_effect <= 1:
        raise ValueError(f"min_detectable_effect must be in (0, 1], got {min_detectable_effect}")

    # Z-scores
    z_alpha = _norm_ppf(1 - alpha / 2)  # two-sided
    z_beta = _norm_ppf(power)

    p1 = baseline_rate
    p2 = baseline_rate + min_detectable_effect

    # Pooled and unpooled variance
    p_bar = (p1 + p2) / 2.0

    n = (
        (z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
         + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        / (min_detectable_effect ** 2)
    )

    return max(1, math.ceil(n))


# ---------------------------------------------------------------------------
# Confidence interval (Wilson score)
# ---------------------------------------------------------------------------

def confidence_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    More accurate than the standard Wald interval, especially for small
    sample sizes or proportions near 0 or 1.

    Args:
        successes: Number of successes.
        trials: Total number of trials.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        (lower_bound, upper_bound) in [0, 1].
    """
    if trials == 0:
        return 0.0, 1.0

    alpha = 1 - confidence
    z = _norm_ppf(1 - alpha / 2)

    p_hat = successes / trials
    denominator = 1 + z**2 / trials

    center = (p_hat + z**2 / (2 * trials)) / denominator
    margin = (z / denominator) * math.sqrt(
        p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2)
    )

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower, upper


# ---------------------------------------------------------------------------
# Internal distribution helpers (stdlib-only approximations)
# ---------------------------------------------------------------------------

def _chi2_sf(x: float, df: float) -> float:
    """Survival function (1 - CDF) for chi-squared distribution.

    For df=1, uses the relationship chi2(1) = Z² where Z ~ N(0,1).
    For df=2, chi2(2) = Exponential(0.5) so SF = exp(-x/2).
    Otherwise falls back to the incomplete gamma series.
    """
    if x <= 0:
        return 1.0
    if df == 1:
        # chi2(1) is the square of a standard normal
        # P(chi2 > x) = 2 * P(Z > sqrt(x))
        return 2.0 * _norm_sf(math.sqrt(x))
    if df == 2:
        # chi2(2) is exponential with rate 1/2
        return math.exp(-x / 2.0)
    return 1.0 - _gammainc(df / 2.0, x / 2.0)


def _gammainc(a: float, x: float, max_iter: int = 200) -> float:
    """Regularized lower incomplete gamma function P(a, x).

    Series expansion: P(a, x) = (x^a / Gamma(a)) * sum_{k=0}^inf (x^k / (a)_{k+1})
    where (a)_k is the rising factorial.
    """
    if x < 0:
        return 0.0
    if a <= 0:
        return 1.0 if x > 0 else 0.0
    if x == 0:
        return 0.0

    log_gamma_a = math.lgamma(a)

    # Series: exp(-x + a*log(x) - log_gamma_a) * sum
    term = math.exp(a * math.log(x) - x - log_gamma_a)
    series_sum = term / a
    for n in range(1, max_iter):
        term *= x / (a + n)
        series_sum += term / (a + n)
        if abs(term / (a + n)) < 1e-15:
            break

    return series_sum


def _t_sf(t: float, df: float) -> float:
    """Survival function for Student's t-distribution.

    Uses the regularized incomplete beta function transformation:
    P(T > t) = 0.5 * I_{df/(df+t^2)}(df/2, 1/2)
    """
    if t < 0:
        return 1.0 - _t_sf(-t, df)
    x = df / (df + t * t)
    return 0.5 * _betainc(df / 2.0, 0.5, x)


def _betainc(a: float, b: float, x: float, max_iter: int = 200) -> float:
    """Regularized incomplete beta function I_x(a, b).

    Uses the continued fraction representation (Lentz's method).
    """
    if x < 0 or x > 1:
        if x < 0:
            return 0.0
        return 1.0
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0

    # Compute via continued fraction
    log_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - log_beta) / a

    # Lentz's continued fraction
    f = 1.0
    c = 1.0
    d = 0.0
    for m in range(1, max_iter):
        # d_{2m}
        d = 1.0 / (1.0 + m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)) * d)
        c = 1.0 + m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)) / max(c, 1e-30)
        f *= c
        f *= d
        # d_{2m+1}
        d = 1.0 / (1.0 - (a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1)) * d)
        c = 1.0 - (a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1)) / max(c, 1e-30)
        f *= c
        f *= d
        if abs(f - 1.0) < 1e-15:
            break

    return min(1.0, max(0.0, front * (f - 1.0)))


def _norm_sf(z: float) -> float:
    """Survival function for standard normal distribution.

    Uses the Abramowitz and Stegun approximation (error < 7.5e-8).
    """
    if z < 0:
        return 1.0 - _norm_sf(-z)
    # Constants for approximation
    p = 0.2316419
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429

    t = 1.0 / (1.0 + p * z)
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    return pdf * (b1 * t + b2 * t**2 + b3 * t**3 + b4 * t**4 + b5 * t**5)


def _norm_ppf(p: float) -> float:
    """Percent point function (inverse CDF) for standard normal.

    Uses the Beasley-Springer-Moro approximation.
    """
    if p <= 0:
        return float("-inf")
    if p >= 1:
        return float("inf")
    if p == 0.5:
        return 0.0

    # Rational approximation
    q = min(p, 1 - p)
    if q < 1e-20:
        return float("inf") if p > 0.5 else float("-inf")

    r = math.sqrt(-2.0 * math.log(q))
    a0 = 2.515517
    a1 = 0.802853
    a2 = 0.010328
    b1 = 1.432788
    b2 = 0.189269
    b3 = 0.001308

    z = r - (a0 + a1 * r + a2 * r**2) / (1.0 + b1 * r + b2 * r**2 + b3 * r**3)

    if p < 0.5:
        z = -z

    return z
