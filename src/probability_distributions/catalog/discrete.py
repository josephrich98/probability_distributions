"""Univariate discrete distributions."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from .._spec import INF, DistSpec, Param, register, register_alias, same


def _prob(name="p", default=0.5, description="probability of success", lower_inclusive=True):
    return Param(name, name, description, default, lower=0, upper=1,
                 lower_inclusive=lower_inclusive, upper_inclusive=True)


def _geometric_domain(P):
    # MGFs of geometric/negative binomial exist for t < −ln(1 − p)
    return (-INF, INF if P["p"] == 1 else -math.log1p(-P["p"]))


def _normalize_probabilities(P):
    p = np.asarray(P["p"], dtype=float)
    total = p.sum()
    if total > 1 + 1e-9:
        raise ValueError(f"probabilities in p sum to {total:g} > 1")
    if total < 1 - 1e-9:
        # Like the notebook: the last category takes the remaining probability
        p = np.append(p, 1 - total)
    return {**P, "p": p / p.sum()}


register(DistSpec(
    name="bernoulli",
    display="Bernoulli",
    kind="discrete",
    params=(_prob(description="probability of success (X = 1)"),),
    scipy=lambda p: stats.bernoulli(p),
    parameter_space="p ∈ [0, 1]",
    support="x ∈ {0, 1}",
    density="P(X = x) = p^x (1 − p)^(1 − x)    i.e. P(X = 1) = p, P(X = 0) = 1 − p",
    cdf="F(x) = 0 for x < 0;  1 − p for 0 ≤ x < 1;  1 for x ≥ 1",
    mean="p",
    variance="p(1 − p)",
    mgf="M(t) = 1 − p + p·e^t",
    story=(
        "A single trial that succeeds (X = 1) with probability p and fails (X = 0) otherwise, e.g. one flip of a "
        "possibly biased coin."
    ),
    use_cases=[
        "Indicator functions: X = I(A) ~ Bern(P(A)).",
        "Response distribution / link function in GLMs (logistic regression).",
        "Any single yes/no, success/failure trial.",
    ],
    special_facts=[
        "Fundamental bridge: for X = I(A), X ~ Bern(P(A)) and E[X] = P(A).",
        "Bern(p) = Binomial(n=1, p).",
        "The sum of n independent Bern(p) is Binomial(n, p).",
        "Categorical with k = 2 categories.",
    ],
    aliases=("bern",),
    mgf_fn=lambda P, t: 1 - P["p"] + P["p"] * np.exp(t),
    style="bar",
))

register(DistSpec(
    name="binomial",
    display="Binomial",
    kind="discrete",
    params=(
        Param("n", "n", "number of trials", 10, kind="int", lower=1, lower_inclusive=True),
        _prob(description="probability of success on each trial"),
    ),
    scipy=lambda n, p: stats.binom(n, p),
    parameter_space="n ∈ {1, 2, …},  p ∈ [0, 1]",
    support="x ∈ {0, 1, …, n}",
    density="P(X = x) = C(n, x) p^x (1 − p)^(n − x)",
    cdf="F(x) = Σ_{i=0}^{⌊x⌋} C(n, i) p^i (1 − p)^(n − i) = I_{1−p}(n − ⌊x⌋, ⌊x⌋ + 1)  (regularized incomplete beta)",
    mean="np",
    variance="np(1 − p)",
    mgf="M(t) = (1 − p + p·e^t)^n",
    story=(
        "Run n independent trials that each succeed with probability p and count the successes, e.g. the number "
        "of heads in n coin flips. Every trial has the same p, as when drawing with replacement."
    ),
    use_cases=[
        "Number of successes in a fixed number of independent trials (conversions, defective parts, votes).",
        "Polling and A/B testing.",
    ],
    special_facts=[
        "Extension of Bernoulli: the sum of n independent Bern(p).",
        "Multinomial with k = 2 categories.",
        "Approaches Normal(np, np(1 − p)) for large n.",
        "Approaches Poisson(λ = np) as n → ∞, p → 0 with np fixed.",
        "Bin(n, p) + Bin(m, p) = Bin(n + m, p) for independent variables with the same p.",
    ],
    aliases=("binom",),
    mgf_fn=lambda P, t: (1 - P["p"] + P["p"] * np.exp(t)) ** P["n"],
))


def _check_hypergeometric(P):
    if P["K"] > P["N"]:
        raise ValueError(f"K={P['K']} must be ≤ N={P['N']}")
    if P["n"] > P["N"]:
        raise ValueError(f"n={P['n']} must be ≤ N={P['N']}")


register(DistSpec(
    name="hypergeometric",
    display="Hypergeometric",
    kind="discrete",
    params=(
        Param("N", "N", "population size", 50, kind="int", lower=1, lower_inclusive=True),
        Param("K", "K", "number of successes in the population", 20, kind="int",
              lower=0, lower_inclusive=True),
        Param("n", "n", "number of draws (without replacement)", 10, kind="int", lower=1, lower_inclusive=True),
    ),
    scipy=lambda N, K, n: stats.hypergeom(M=N, n=K, N=n),
    parameter_space="N ∈ {1, 2, …},  K ∈ {0, 1, …, N},  n ∈ {1, 2, …, N}",
    support="x ∈ {max(0, n + K − N), …, min(n, K)}",
    density="P(X = x) = C(K, x) C(N − K, n − x) / C(N, n)",
    cdf="F(x) = Σ_{i ≤ x} P(X = i)   (no closed form – use the definition)",
    mean="n K / N",
    variance="n (K/N) (1 − K/N) (N − n)/(N − 1)",
    mgf="No closed form – use the definition M(t) = Σ_x e^(tx) P(X = x)",
    story=(
        "Draw n items without replacement from a population of N items, K of which count as successes, and count "
        "the successes drawn. E.g. a herd of N elk contains K tagged elk; capture n of them and count the tagged "
        "ones."
    ),
    use_cases=[
        "Capture–recapture estimates of population size.",
        "Card games (e.g. number of aces in a hand).",
        "Fisher's exact test; gene-set enrichment (over-representation) analysis.",
        "Quality control: defective items in a sample drawn without replacement.",
    ],
    special_facts=[
        "Like Binomial but without replacement.",
        "Approaches Binomial(n, K/N) as N → ∞ with K/N fixed.",
        "(N − n)/(N − 1) is the finite population correction to the binomial variance.",
    ],
    aliases=("hypergeom",),
    check=_check_hypergeometric,
))

register(DistSpec(
    name="geometric",
    display="Geometric",
    kind="discrete",
    params=(_prob(description="probability of success on each trial", lower_inclusive=False),),
    scipy=lambda p: stats.geom(p),
    parameter_space="p ∈ (0, 1]",
    support="x ∈ {1, 2, 3, …}",
    density="P(X = x) = (1 − p)^(x − 1) p",
    cdf="F(x) = 1 − (1 − p)^⌊x⌋",
    mean="1/p",
    variance="(1 − p)/p²",
    mgf="M(t) = p·e^t / (1 − (1 − p)e^t),  for t < −ln(1 − p)",
    story=(
        "Repeat independent trials, each succeeding with probability p, until the first success; X is the number "
        "of trials needed, counting the successful one. (See 'geometric_failures' for the version that counts "
        "only the failures.)"
    ),
    use_cases=[
        "Number of attempts until the first success (sales calls, login attempts, die rolls until a 6).",
    ],
    special_facts=[
        "Discrete analog of the exponential: the only memoryless discrete distribution, "
        "P(X > s + t | X > s) = P(X > t).",
        "X − 1 ~ Geometric(failures) = NegBin(r = 1, p).",
    ],
    aliases=("geom", "geometric_trials"),
    mgf_fn=lambda P, t: P["p"] * np.exp(t) / (1 - (1 - P["p"]) * np.exp(t)),
    mgf_domain=_geometric_domain,
))

register(DistSpec(
    name="geometric_failures",
    display="Geometric (failures)",
    kind="discrete",
    params=(_prob(description="probability of success on each trial", lower_inclusive=False),),
    scipy=lambda p: stats.geom(p, loc=-1),
    parameter_space="p ∈ (0, 1]",
    support="x ∈ {0, 1, 2, …}",
    density="P(X = x) = (1 − p)^x p",
    cdf="F(x) = 1 − (1 − p)^(⌊x⌋ + 1)",
    mean="(1 − p)/p",
    variance="(1 − p)/p²",
    mgf="M(t) = p / (1 − (1 − p)e^t),  for t < −ln(1 − p)",
    story=(
        "Repeat independent trials, each succeeding with probability p, until the first success; X is the number "
        "of failures before it."
    ),
    use_cases=[
        "Number of failures before the first success.",
    ],
    special_facts=[
        "NegBin(r = 1, p).",
        "Discrete analog of the exponential (memoryless).",
        "X + 1 ~ Geometric(trials).",
        "The sum of r independent Geometric(failures, p) is NegBin(r, p).",
    ],
    aliases=("geometric_excl", "geometric0"),
    mgf_fn=lambda P, t: P["p"] / (1 - (1 - P["p"]) * np.exp(t)),
    mgf_domain=_geometric_domain,
))

register(DistSpec(
    name="negative_binomial",
    display="Negative Binomial",
    kind="discrete",
    params=(
        Param("r", "r", "number of successes to wait for (any r > 0 is allowed)", 5, lower=0),
        _prob(description="probability of success on each trial", lower_inclusive=False),
    ),
    scipy=lambda r, p: stats.nbinom(r, p),
    parameter_space="r > 0 (an integer in the story; any positive real is allowed),  p ∈ (0, 1]",
    support="x ∈ {0, 1, 2, …}",
    density="P(X = x) = C(x + r − 1, x) (1 − p)^x p^r",
    cdf="F(x) = I_p(r, ⌊x⌋ + 1)  (regularized incomplete beta)",
    mean="r(1 − p)/p",
    variance="r(1 − p)/p²",
    mgf="M(t) = (p / (1 − (1 − p)e^t))^r,  for t < −ln(1 − p)",
    story=(
        "Repeat independent trials, each succeeding with probability p, until the r-th success; X is the number "
        "of failures along the way."
    ),
    use_cases=[
        "Overdispersed Poisson: empirically used for count data where variance > mean "
        "(RNA-seq read counts, insurance claims, ecology).",
    ],
    special_facts=[
        "Extension of the geometric: the sum of r independent Geometric(failures, p).",
        "Variance > mean (always).",
        "Poisson–Gamma mixture: if λ ~ Gamma(r, p/(1 − p)) and X | λ ~ Poisson(λ), then X ~ NegBin(r, p).",
        "Approaches Poisson as r → ∞ with the mean r(1 − p)/p held fixed.",
    ],
    aliases=("negbin", "nbinom", "negative_binomial_failures"),
    mgf_fn=lambda P, t: (P["p"] / (1 - (1 - P["p"]) * np.exp(t))) ** P["r"],
    mgf_domain=_geometric_domain,
))

register(DistSpec(
    name="poisson",
    display="Poisson",
    kind="discrete",
    params=(Param("lam", "λ", "expected number of events in the window (rate × window length)", 4.0, lower=0),),
    scipy=lambda lam: stats.poisson(lam),
    parameter_space="λ > 0",
    support="x ∈ {0, 1, 2, …}",
    density="P(X = x) = λ^x e^(−λ) / x!",
    cdf="F(x) = e^(−λ) Σ_{i=0}^{⌊x⌋} λ^i / i! = Q(⌊x⌋ + 1, λ)  (regularized upper incomplete gamma)",
    mean="λ",
    variance="λ",
    mgf="M(t) = exp(λ(e^t − 1))",
    story=(
        "Count the events in a fixed window of time or space when events happen independently at a constant "
        "average rate, e.g. emails per hour or chocolate chips per cookie. It is the count of many rare, "
        "independent opportunities for an event."
    ),
    use_cases=[
        "Counts: arrivals, decays, mutations, typos per page, calls per hour.",
        "Poisson regression for count data.",
    ],
    special_facts=[
        "Limit of Binomial(n, p) as n → ∞, p → 0 with np = λ fixed.",
        "Variance = mean = λ.",
        "Poisson(λ₁) + Poisson(λ₂) = Poisson(λ₁ + λ₂) for independent variables.",
        "Waiting times between events in a Poisson process are Exponential(λ).",
        "Approaches Normal(λ, λ) for large λ.",
    ],
    alt_params={"mu": same("lam"), "mean": same("lam")},
    mgf_fn=lambda P, t: np.exp(P["lam"] * (np.exp(t) - 1)),
))


def _check_discrete_uniform(P):
    if P["b"] < P["a"]:
        raise ValueError(f"need a ≤ b, got a={P['a']}, b={P['b']}")


def _discrete_uniform_mgf(P, t):
    a, b = P["a"], P["b"]
    t = np.asarray(t, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = (np.exp(a * t) - np.exp((b + 1) * t)) / ((b - a + 1) * (1 - np.exp(t)))
    return np.where(t == 0, 1.0, value)


register(DistSpec(
    name="discrete_uniform",
    display="Discrete Uniform",
    kind="discrete",
    params=(
        Param("a", "a", "smallest value", 1, kind="int"),
        Param("b", "b", "largest value", 6, kind="int"),
    ),
    scipy=lambda a, b: stats.randint(a, b + 1),
    parameter_space="a, b ∈ ℤ with a ≤ b  (n = b − a + 1 outcomes)",
    support="x ∈ {a, a + 1, …, b}",
    density="P(X = x) = 1/n = 1/(b − a + 1)",
    cdf="F(x) = (⌊x⌋ − a + 1)/(b − a + 1)  for a ≤ x ≤ b",
    mean="(a + b)/2",
    variance="((b − a + 1)² − 1)/12",
    mgf="M(t) = (e^(at) − e^((b+1)t)) / ((b − a + 1)(1 − e^t))  (M(0) = 1)",
    story=(
        "Pick one of the integers a, a + 1, …, b, each equally likely, e.g. roll a fair die (a = 1, b = 6)."
    ),
    use_cases=[
        "Fair dice, random integer generation, uninformative prior on a finite set.",
    ],
    special_facts=[
        "Categorical where each p_i = 1/n.",
        "Multinomial(n = 1) with equal probabilities.",
        "Beta-Binomial(n, α = 1, β = 1) is discrete uniform on {0, …, n}.",
    ],
    aliases=("uniform_discrete", "randint"),
    check=_check_discrete_uniform,
    mgf_fn=_discrete_uniform_mgf,
    style="bar",
))

register(DistSpec(
    name="beta_binomial",
    display="Beta-Binomial",
    kind="discrete",
    params=(
        Param("n", "n", "number of trials", 10, kind="int", lower=1, lower_inclusive=True),
        Param("alpha", "α", "shape α of the Beta distribution of p (larger → p nearer 1)", 2.0, lower=0),
        Param("beta", "β", "shape β of the Beta distribution of p (larger → p nearer 0)", 2.0, lower=0),
    ),
    scipy=lambda n, alpha, beta: stats.betabinom(n, alpha, beta),
    parameter_space="n ∈ {1, 2, …},  α > 0,  β > 0",
    support="x ∈ {0, 1, …, n}",
    density="P(X = x) = C(n, x) B(x + α, n − x + β) / B(α, β)",
    cdf="F(x) = Σ_{i=0}^{⌊x⌋} P(X = i)",
    mean="nα/(α + β)",
    variance="nαβ(α + β + n) / ((α + β)²(α + β + 1))",
    mgf="M(t) = ₂F₁(−n, α; α + β; 1 − e^t)  (Gauss hypergeometric function)",
    story=(
        "First draw a success probability p from Beta(α, β), then count the successes in n trials that each "
        "succeed with that p. The uncertainty in p spreads the counts out more than a binomial."
    ),
    use_cases=[
        "Overdispersed binomial data.",
        "Bayesian models where p varies across experiments (posterior predictive of a Beta–Binomial model).",
    ],
    special_facts=[
        "Overdispersed binomial: variance exceeds the binomial variance np(1 − p) with p = α/(α + β).",
        "α = β = 1 gives the discrete uniform on {0, …, n}.",
        "Approaches Binomial(n, p) as α, β → ∞ with α/(α + β) = p fixed.",
        "n = 1 gives Bernoulli(α/(α + β)).",
    ],
    aliases=("betabinom", "betabinomial"),
    alt_params={"a": same("alpha"), "b": same("beta")},
))


def _categorical_xrange(P, frozen):
    return (1, len(P["p"]))


register(DistSpec(
    name="categorical",
    display="Categorical",
    kind="discrete",
    params=(Param("p", "p", "probabilities of the k categories (the last is filled in if they sum to < 1)",
                  [0.2, 0.5, 0.3], kind="vector", lower=0, upper=1, lower_inclusive=True, upper_inclusive=True),),
    scipy=lambda p: stats.rv_discrete(values=(np.arange(1, len(p) + 1), p)),
    parameter_space="p = (p_1, …, p_k) with p_i ≥ 0 and Σ p_i = 1",
    support="x ∈ {1, 2, …, k}  (category labels)",
    density="P(X = i) = p_i",
    cdf="F(x) = Σ_{i ≤ x} p_i",
    mean="Σ i·p_i  (only meaningful if the category labels 1…k are numeric)",
    variance="Σ i²·p_i − (Σ i·p_i)²  (only meaningful for numeric labels)",
    mgf="M(t) = Σ_{i=1}^{k} p_i e^(it)",
    story=(
        "A single draw from k categories where category i has probability p_i, e.g. one roll of a loaded k-sided "
        "die."
    ),
    use_cases=[
        "Classification outputs (softmax), word/token choice, any single draw from k categories.",
    ],
    special_facts=[
        "Also called multinoulli: the multivariate (one-hot) generalization of the Bernoulli.",
        "Multinomial with n = 1.",
        "Generalization of the discrete uniform (all p_i = 1/k).",
        "k = 2 is a (relabeled) Bernoulli.",
        "Conjugate prior: Dirichlet.",
    ],
    aliases=("multinoulli", "cat"),
    normalize=_normalize_probabilities,
    mgf_fn=lambda P, t: np.sum(P["p"][:, None] * np.exp(np.outer(np.arange(1, len(P["p"]) + 1), np.atleast_1d(t))),
                               axis=0).reshape(np.shape(t)),
    xrange=_categorical_xrange,
    style="bar",
))

register_alias("die", "categorical", p=[1 / 6] * 6)
