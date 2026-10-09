"""Multivariate distributions (plotted as 3-D surfaces/bars for k = 3 categories or 2 dimensions)."""

from __future__ import annotations

import numpy as np
from scipy import stats

from .._spec import DistSpec, Param, register, same
from .discrete import _normalize_probabilities


def _multinomial_cov(P):
    n, p = P["n"], P["p"]
    return n * (np.diag(p) - np.outer(p, p))


register(DistSpec(
    name="multinomial",
    display="Multinomial",
    kind="discrete",
    params=(
        Param("n", "n", "number of draws", 10, kind="int", lower=1, lower_inclusive=True),
        Param("p", "p", "probabilities of the k categories (the last is filled in if they sum to < 1)",
              [0.3, 0.4, 0.3], kind="vector", lower=0, upper=1, lower_inclusive=True, upper_inclusive=True),
    ),
    scipy=lambda n, p: stats.multinomial(n, p),
    parameter_space="n ∈ {1, 2, …},  p = (p_1, …, p_k) with p_i ≥ 0 and Σ p_i = 1",
    support="x = (x_1, …, x_k) with x_i ∈ {0, 1, …, n} and Σ x_i = n",
    density="P(X = x) = n!/(x_1! ⋯ x_k!) · p_1^(x_1) ⋯ p_k^(x_k)",
    cdf="No closed form",
    mean="E[X_i] = n p_i",
    variance="Var(X_i) = n p_i (1 − p_i),  Cov(X_i, X_j) = −n p_i p_j",
    mgf="M(t) = (Σ_i p_i e^(t_i))^n",
    story=(
        "Make n independent draws from k categories with probabilities p_1, …, p_k and tally how many land in "
        "each category, e.g. roll a loaded k-sided die n times and count each face."
    ),
    use_cases=[
        "Counts in k categories: word counts (bag of words), allele counts, survey responses, histogram bins.",
    ],
    special_facts=[
        "Multivariate binomial (k > 2) and multivariate categorical (n > 1): k = 2 is the Binomial, "
        "n = 1 is the Categorical.",
        "Each marginal X_i ~ Binomial(n, p_i).",
        "Can be visualized in k − 1 dimensions since the x_i sum to n.",
        "Conjugate prior: Dirichlet.",
    ],
    normalize=_normalize_probabilities,
    multivariate=True,
    mgf_fn=lambda P, t: (np.exp(np.asarray(t, dtype=float)) @ P["p"]) ** P["n"],
    mean_fn=lambda P: P["n"] * P["p"],
    var_fn=_multinomial_cov,
))


def _check_mvn(P):
    mean, cov = P["mu"], P["cov"]
    if cov.shape != (len(mean), len(mean)):
        raise ValueError(f"cov must be {len(mean)}×{len(mean)} to match mean, got {cov.shape[0]}×{cov.shape[1]}")
    if not np.allclose(cov, cov.T):
        raise ValueError("cov must be symmetric")
    if np.min(np.linalg.eigvalsh(cov)) <= 0:
        raise ValueError("cov must be positive definite")


register(DistSpec(
    name="multivariate_normal",
    display="Multivariate Normal",
    kind="continuous",
    params=(
        Param("mu", "μ", "mean vector", [0.0, 0.0], kind="vector"),
        Param("cov", "Σ", "covariance matrix (symmetric positive definite)", [[1.0, 0.5], [0.5, 1.0]], kind="matrix"),
    ),
    scipy=lambda mu, cov: stats.multivariate_normal(mu, cov),
    parameter_space="μ ∈ ℝ^k,  Σ ∈ ℝ^(k×k) symmetric positive definite",
    support="x ∈ ℝ^k",
    density="f(x) = (2π)^(−k/2) |Σ|^(−1/2) exp(−½ (x − μ)ᵀ Σ⁻¹ (x − μ))",
    cdf="No closed form (computed numerically)",
    mean="μ",
    variance="Σ  (Cov(X_i, X_j) = Σ_ij;  for k = 2, Σ = [[σ_x², ρσ_xσ_y], [ρσ_xσ_y, σ_y²]])",
    mgf="M(t) = exp(μᵀt + ½ tᵀΣt)",
    story=(
        "Several measurements that are each normal and linearly related, e.g. height and weight; every linear "
        "combination of them is normal."
    ),
    use_cases=[
        "Correlated measurements, Gaussian processes, Kalman filters, linear/quadratic discriminant analysis.",
    ],
    special_facts=[
        "Every linear combination aᵀX is (univariate) Normal; marginals and conditionals are Normal.",
        "For jointly normal variables, uncorrelated ⇔ independent.",
        "(X − μ)ᵀ Σ⁻¹ (X − μ) ~ χ²(k) (squared Mahalanobis distance).",
    ],
    aliases=("mvn", "multivariate_gaussian", "bivariate_normal"),
    alt_params={"mean": same("mu"), "Sigma": same("cov")},
    check=_check_mvn,
    multivariate=True,
    mgf_fn=lambda P, t: np.exp(np.asarray(t, dtype=float) @ P["mu"]
                               + 0.5 * np.einsum("...i,ij,...j->...", np.asarray(t, dtype=float), P["cov"],
                                                 np.asarray(t, dtype=float))),
    mean_fn=lambda P: P["mu"],
    var_fn=lambda P: P["cov"],
))


def _dirichlet_cov(P):
    a = P["alpha"]
    a0 = a.sum()
    m = a / a0
    return (np.diag(m) - np.outer(m, m)) / (a0 + 1)


def _check_dirichlet(P):
    if len(P["alpha"]) < 2:
        raise ValueError("alpha needs at least 2 entries")


register(DistSpec(
    name="dirichlet",
    display="Dirichlet",
    kind="continuous",
    params=(Param("alpha", "α", "concentrations α_1, …, α_k (≈ pseudo-counts per category)", [2.0, 2.0, 2.0], kind="vector", lower=0),),
    scipy=lambda alpha: stats.dirichlet(alpha),
    parameter_space="α = (α_1, …, α_k) with k ≥ 2 and α_i > 0  (α_0 = Σ α_i)",
    support="x = (x_1, …, x_k) with x_i ∈ [0, 1] and Σ x_i = 1  (the probability simplex)",
    density="f(x) = Π x_i^(α_i − 1) / B(α),  B(α) = Π Γ(α_i) / Γ(α_0)",
    cdf="No closed form",
    mean="E[X_i] = α_i/α_0",
    variance="Var(X_i) = α_i(α_0 − α_i)/(α_0²(α_0 + 1)),  Cov(X_i, X_j) = −α_iα_j/(α_0²(α_0 + 1))",
    mgf="No simple closed form",
    story=(
        "A random probability vector (k nonnegative numbers that sum to 1), e.g. the unknown face probabilities "
        "of a die. α_i acts like a pseudo-count of observations in category i."
    ),
    use_cases=[
        "Bayesian models of proportions, topic modeling (LDA), compositional data.",
        "Conjugate prior for the Categorical/Multinomial.",
    ],
    special_facts=[
        "Multivariate Beta: k = 2 gives X_1 ~ Beta(α_1, α_2).",
        "If Y_i ~ Gamma(α_i, 1) independent, then (Y_1, …, Y_k)/Σ Y_j ~ Dirichlet(α).",
        "All α_i = 1 is uniform over the simplex.",
        "Aggregation: merging components sums their α's.",
    ],
    check=_check_dirichlet,
    multivariate=True,
    mean_fn=lambda P: P["alpha"] / P["alpha"].sum(),
    var_fn=_dirichlet_cov,
))
