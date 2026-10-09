"""Univariate continuous distributions."""

from __future__ import annotations

import numpy as np
from scipy import special, stats

from .._spec import INF, DistSpec, Param, inverse, register, register_alias, same

def _positive(name, symbol, description, default):
    return Param(name, symbol, description, default, lower=0)


def _real(name, symbol, description, default):
    return Param(name, symbol, description, default)


def _check_uniform(P):
    if P["b"] <= P["a"]:
        raise ValueError(f"need a < b, got a={P['a']}, b={P['b']}")


def _uniform_mgf(P, t):
    a, b = P["a"], P["b"]
    t = np.asarray(t, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = (np.exp(t * b) - np.exp(t * a)) / (t * (b - a))
    return np.where(t == 0, 1.0, value)


register(DistSpec(
    name="uniform",
    display="Uniform",
    kind="continuous",
    params=(_real("a", "a", "lower bound", 0.0), _real("b", "b", "upper bound", 1.0)),
    scipy=lambda a, b: stats.uniform(loc=a, scale=b - a),
    parameter_space="a, b ∈ ℝ with a < b",
    support="x ∈ [a, b]",
    density="f(x) = 1/(b − a)",
    cdf="F(x) = (x − a)/(b − a)  for a ≤ x ≤ b",
    mean="(a + b)/2",
    variance="(b − a)²/12",
    mgf="M(t) = (e^(tb) − e^(ta)) / (t(b − a))  (M(0) = 1)",
    story=(
        "A point chosen completely at random from the interval [a, b]: any two sub-intervals of the same length "
        "are equally likely to contain it."
    ),
    use_cases=[
        "Random number generation, uninformative priors on bounded parameters, rounding error.",
    ],
    special_facts=[
        "Uniform(0, 1) = Beta(1, 1).",
        "Universality of the uniform: if U ~ Unif(0, 1) then F⁻¹(U) ~ F (inverse transform sampling), "
        "and F(X) ~ Unif(0, 1) for any continuous X.",
        "−ln(U)/λ ~ Exponential(λ).",
        "Order statistics of n Unif(0, 1) variables are Beta distributed: U_(j) ~ Beta(j, n − j + 1).",
    ],
    aliases=("continuous_uniform", "unif"),
    check=_check_uniform,
    mgf_fn=_uniform_mgf,
    xrange=lambda P, d: (P["a"] - 0.1 * (P["b"] - P["a"]), P["b"] + 0.1 * (P["b"] - P["a"])),
))

register(DistSpec(
    name="normal",
    display="Normal",
    kind="continuous",
    params=(_real("mu", "μ", "mean", 0.0), _positive("sigma", "σ", "standard deviation", 1.0)),
    scipy=lambda mu, sigma: stats.norm(mu, sigma),
    parameter_space="μ ∈ ℝ,  σ > 0",
    support="x ∈ ℝ",
    density="f(x) = 1/(σ√(2π)) · exp(−(x − μ)²/(2σ²))",
    cdf="F(x) = Φ((x − μ)/σ) = ½[1 + erf((x − μ)/(σ√2))]",
    mean="μ",
    variance="σ²",
    mgf="M(t) = exp(μt + σ²t²/2)",
    story=(
        "The sum (or average) of many small, independent effects, e.g. measurement error or height. The central "
        "limit theorem makes it the limit of averages of almost anything."
    ),
    use_cases=[
        "Measurement error, heights, test scores, sampling distributions of means (CLT).",
        "Linear regression errors; Gaussian processes; Kalman filters.",
    ],
    special_facts=[
        "Central limit theorem: the (standardized) mean of many iid draws from any distribution with finite "
        "variance is approximately Normal.",
        "Conjugate prior for itself (for the mean of a Normal with known variance).",
        "Standardization: Z = (X − μ)/σ ~ N(0, 1); Z² ~ χ²(1).",
        "Sums of independent Normals are Normal: N(μ₁, σ₁²) + N(μ₂, σ₂²) = N(μ₁ + μ₂, σ₁² + σ₂²).",
        "e^X ~ Lognormal(μ, σ).",
        "Maximum-entropy distribution for a given mean and variance.",
        "68–95–99.7 rule: P(|X − μ| < kσ) ≈ 0.68, 0.95, 0.997 for k = 1, 2, 3.",
    ],
    aliases=("gaussian", "norm"),
    alt_params={"mean": same("mu"), "loc": same("mu"), "sd": same("sigma"), "std": same("sigma"),
                "scale": same("sigma"),
                "variance": ("sigma", lambda v: v ** 0.5, "variance; sigma = √variance")},
    mgf_fn=lambda P, t: np.exp(P["mu"] * t + P["sigma"] ** 2 * t ** 2 / 2),
    xrange=lambda P, d: (P["mu"] - 4 * P["sigma"], P["mu"] + 4 * P["sigma"]),
))
register_alias("standard_normal", "normal", mu=0.0, sigma=1.0)

register(DistSpec(
    name="exponential",
    display="Exponential",
    kind="continuous",
    params=(_positive("lam", "λ", "rate (events per unit time); mean = 1/λ", 1.0),),
    scipy=lambda lam: stats.expon(scale=1 / lam),
    parameter_space="λ > 0  (rate; the scale is 1/λ)",
    support="x ∈ [0, ∞)",
    density="f(x) = λ e^(−λx)",
    cdf="F(x) = 1 − e^(−λx)",
    mean="1/λ",
    variance="1/λ²",
    mgf="M(t) = λ/(λ − t),  for t < λ",
    story=(
        "The waiting time until the next event when events happen independently at a constant rate λ (a Poisson "
        "process). No matter how long you have already waited, the remaining wait has the same distribution."
    ),
    use_cases=[
        "Waiting times, lifetimes of components without wear, radioactive decay, survival analysis baseline.",
    ],
    special_facts=[
        "Continuous analog of the geometric: the only memoryless continuous distribution, "
        "P(X > s + t | X > s) = P(X > t).",
        "Exp(λ) = Gamma(α = 1, β = λ) = Weibull(k = 1, λ_scale = 1/λ); χ²(2) = Exp(1/2).",
        "The minimum of independent Exp(λ_i) is Exp(Σ λ_i).",
        "The sum of α independent Exp(λ) is Gamma(α, λ).",
        "The difference of two iid Exp(λ) is Laplace(0, 1/λ).",
        "If U ~ Unif(0, 1), −ln(U)/λ ~ Exp(λ); conversely e^(−λX) ~ Unif(0, 1).",
        "Powers of exponentials are Weibull: if X ~ Exp(1), cX^(1/k) ~ Weibull(k, c).",
    ],
    aliases=("exp", "expon"),
    alt_params={"rate": same("lam"), "scale": inverse("lam", "scale θ"),
                "mean": inverse("lam", "scale θ")},
    mgf_fn=lambda P, t: P["lam"] / (P["lam"] - t),
    mgf_domain=lambda P: (-INF, P["lam"]),
))

register(DistSpec(
    name="gamma",
    display="Gamma",
    kind="continuous",
    params=(
        _positive("alpha", "α", "shape (number of events waited for)", 2.0),
        _positive("beta", "β", "rate (events per unit time); scale θ = 1/β", 1.0),
    ),
    scipy=lambda alpha, beta: stats.gamma(alpha, scale=1 / beta),
    parameter_space="α > 0 (shape),  β > 0 (rate)",
    support="x ∈ (0, ∞)",
    density="f(x) = β^α / Γ(α) · x^(α − 1) e^(−βx)",
    cdf="F(x) = γ(α, βx)/Γ(α) = P(α, βx)  (regularized lower incomplete gamma)",
    mean="α/β",
    variance="α/β²",
    mgf="M(t) = (1 − t/β)^(−α),  for t < β",
    story=(
        "The waiting time until the α-th event when events happen independently at rate β (a Poisson process), "
        "i.e. the sum of α independent Exponential(β) waiting times. α need not be a whole number."
    ),
    use_cases=[
        "Waiting times, rainfall amounts, insurance claim sizes.",
        "Bayesian conjugate prior for a Poisson rate and for a Normal precision.",
    ],
    special_facts=[
        "Continuous analog of the negative binomial.",
        "Generalizes the exponential: Gamma(1, β) = Exp(β).",
        "Gamma(k/2, 1/2) = χ²(k).",
        "Gamma(α₁, β) + Gamma(α₂, β) = Gamma(α₁ + α₂, β) for independent variables.",
        "If X ~ Gamma(α, β), Y ~ Gamma(α', β) independent: X/(X + Y) ~ Beta(α, α').",
        "Approaches Normal(α/β, α/β²) for large α.",
        "Shape–scale form: k = α and θ = 1/β (mean kθ, variance kθ²); pass k= and theta= to use it.",
    ],
    alt_params={"shape": same("alpha"), "k": same("alpha"), "a": same("alpha"), "rate": same("beta"),
                "scale": inverse("beta", "scale θ"), "theta": inverse("beta", "scale θ")},
    mgf_fn=lambda P, t: (1 - t / P["beta"]) ** (-P["alpha"]),
    mgf_domain=lambda P: (-INF, P["beta"]),
))

register(DistSpec(
    name="beta",
    display="Beta",
    kind="continuous",
    params=(_positive("alpha", "α", "shape (larger → mass pushed toward 1)", 2.0),
            _positive("beta", "β", "shape (larger → mass pushed toward 0)", 5.0)),
    scipy=lambda alpha, beta: stats.beta(alpha, beta),
    parameter_space="α > 0,  β > 0",
    support="x ∈ [0, 1]",
    density="f(x) = x^(α − 1) (1 − x)^(β − 1) / B(α, β)",
    cdf="F(x) = I_x(α, β)  (regularized incomplete beta)",
    mean="α/(α + β)",
    variance="αβ / ((α + β)²(α + β + 1))",
    mgf="M(t) = ₁F₁(α; α + β; t) = 1 + Σ_{k≥1} (Π_{r=0}^{k−1} (α + r)/(α + β + r)) t^k/k!",
    story=(
        "A random probability or proportion on [0, 1]. Starting from a uniform prior on a success probability and"
        " observing α − 1 successes and β − 1 failures gives a Beta(α, β) posterior. It is also the j-th smallest"
        " of n uniform random numbers (α = j, β = n − j + 1)."
    ),
    use_cases=[
        "Modeling a probability (support is [0, 1]), e.g. a Bayesian prior for a success rate.",
        "Proportions and fractions; PERT task-duration estimates.",
    ],
    special_facts=[
        "Conjugate prior for the Bernoulli/Binomial (and Geometric/NegBin) p: "
        "Beta(α, β) prior + x successes, y failures → Beta(α + x, β + y) posterior.",
        "Beta(1, 1) = Uniform(0, 1).",
        "If X ~ Gamma(α, c), Y ~ Gamma(β, c) independent: X/(X + Y) ~ Beta(α, β).",
        "Order statistics of uniforms: U_(j) of n ~ Beta(j, n − j + 1).",
        "Dirichlet with k = 2 categories.",
        "Approaches Normal for large α and β.",
    ],
    alt_params={"a": same("alpha"), "b": same("beta")},
    mgf_fn=lambda P, t: special.hyp1f1(P["alpha"], P["alpha"] + P["beta"], t),
    xrange=lambda P, d: (0.0, 1.0),
))

register(DistSpec(
    name="weibull",
    display="Weibull",
    kind="continuous",
    params=(_positive("k", "k", "shape (k < 1: failure rate falls with age, k > 1: rises)", 1.5),
            _positive("lam", "λ", "scale (characteristic lifetime: 63.2% fail by λ)", 1.0)),
    scipy=lambda k, lam: stats.weibull_min(k, scale=lam),
    parameter_space="k > 0 (shape),  λ > 0 (scale)",
    support="x ∈ [0, ∞)",
    density="f(x) = (k/λ)(x/λ)^(k − 1) e^(−(x/λ)^k)",
    cdf="F(x) = 1 − e^(−(x/λ)^k)",
    mean="λ Γ(1 + 1/k)",
    variance="λ² [Γ(1 + 2/k) − Γ(1 + 1/k)²]",
    mgf="M(t) = Σ_{n≥0} (tλ)^n/n! · Γ(1 + n/k)  for k > 1 (no closed form; computed numerically)",
    story=(
        "The lifetime of something whose failure rate changes as a power of its age: rising with age (wear-out) "
        "when k > 1, falling (infant mortality) when k < 1, and constant when k = 1."
    ),
    use_cases=[
        "Reliability engineering and survival analysis, wind speeds, material strength.",
    ],
    special_facts=[
        "Generalizes the exponential: Weibull(k = 1, λ) = Exp(rate 1/λ).",
        "k < 1 = decreasing hazard (infant mortality), k = 1 = constant hazard, k > 1 = increasing hazard (wear-out).",
        "If X ~ Exp(1) then λX^(1/k) ~ Weibull(k, λ).",
        "Type III extreme value distribution (for minima).",
    ],
    alt_params={"shape": same("k"), "c": same("k"), "scale": same("lam")},
    mgf_domain=lambda P: (-INF, INF) if P["k"] > 1 else ((-INF, 1 / P["lam"]) if P["k"] == 1 else (-INF, 0.0)),
    xrange=lambda P, d: (0.0, min(d.ppf(0.999), 5 * P["lam"])),
))

register(DistSpec(
    name="pareto",
    display="Pareto",
    kind="continuous",
    params=(_positive("alpha", "α", "tail index (larger → thinner tail)", 2.0),
            _positive("xm", "xₘ", "minimum possible value (scale)", 1.0)),
    scipy=lambda alpha, xm: stats.pareto(alpha, scale=xm),
    parameter_space="α > 0 (shape),  xₘ > 0 (scale)",
    support="x ∈ [xₘ, ∞)",
    density="f(x) = α xₘ^α / x^(α + 1)",
    cdf="F(x) = 1 − (xₘ/x)^α",
    mean="α xₘ/(α − 1) for α > 1 (∞ otherwise)",
    variance="xₘ² α / ((α − 1)²(α − 2)) for α > 2 (∞ otherwise)",
    mgf="Does not exist for t > 0 (heavy tail)",
    story=(
        "A power law: the chance of exceeding x falls off like (xₘ/x)^α above a minimum value xₘ. 'Rich get "
        "richer' processes produce it, e.g. wealth, city populations and file sizes."
    ),
    use_cases=[
        "Income/wealth distributions (the 80–20 rule is α ≈ 1.16), insurance losses, internet traffic.",
    ],
    special_facts=[
        "Heavy-tailed power law; the mean exists only for α > 1 and the variance only for α > 2.",
        "ln(X/xₘ) ~ Exp(α).",
        "Scale-free: conditional on X > c ≥ xₘ, X ~ Pareto(α, c).",
    ],
    alt_params={"b": same("alpha"), "shape": same("alpha"), "scale": same("xm"), "x_m": same("xm")},
    mgf_domain=lambda P: (-INF, 0.0),
    xrange=lambda P, d: (P["xm"], min(d.ppf(0.95), 10 * P["xm"])),
))

register(DistSpec(
    name="cauchy",
    display="Cauchy",
    kind="continuous",
    params=(_real("x0", "x₀", "location (median and mode)", 0.0), _positive("gamma", "γ", "scale (half-width at half-maximum)", 1.0)),
    scipy=lambda x0, gamma: stats.cauchy(x0, gamma),
    parameter_space="x₀ ∈ ℝ,  γ > 0",
    support="x ∈ ℝ",
    density="f(x) = 1 / (πγ [1 + ((x − x₀)/γ)²])",
    cdf="F(x) = ½ + arctan((x − x₀)/γ)/π",
    mean="Undefined",
    variance="Undefined",
    mgf="Does not exist (characteristic function: e^(ix₀t − γ|t|))",
    story=(
        "The ratio of two independent standard normals. Equivalently, shine a laser from a point γ above x₀ at a "
        "uniformly random angle and record where it hits the x-axis."
    ),
    use_cases=[
        "Resonance line shapes in physics (Lorentzian), robust statistics, counterexamples.",
    ],
    special_facts=[
        "Undefined mean and variance; extremely heavy tails — the law of large numbers fails.",
        "Student-t with ν = 1.",
        "The mean of n iid Cauchy(x₀, γ) is again Cauchy(x₀, γ) (stable distribution).",
        "If X ~ Cauchy(0, 1) then 1/X ~ Cauchy(0, 1).",
    ],
    mgf_domain=lambda P: (0.0, 0.0),
    alt_params={"loc": same("x0"), "scale": same("gamma")},
    xrange=lambda P, d: (P["x0"] - 10 * P["gamma"], P["x0"] + 10 * P["gamma"]),
))


register(DistSpec(
    name="student_t",
    display="Student-t",
    kind="continuous",
    params=(_positive("nu", "ν", "degrees of freedom", 3.0),),
    scipy=lambda nu: stats.t(nu),
    parameter_space="ν > 0",
    support="x ∈ ℝ",
    density="f(x) = Γ((ν + 1)/2) / (√(νπ) Γ(ν/2)) · (1 + x²/ν)^(−(ν + 1)/2)",
    cdf="F(x) = 1 − ½ I_{ν/(x² + ν)}(ν/2, ½) for x ≥ 0 (by symmetry F(−x) = 1 − F(x))",
    mean="0 for ν > 1 (undefined otherwise)",
    variance="ν/(ν − 2) for ν > 2;  ∞ for 1 < ν ≤ 2;  undefined otherwise",
    mgf="Does not exist",
    story=(
        "The standardized mean of a normal sample when the standard deviation has to be estimated from the same "
        "sample: (x̄ − μ)/(s/√n) ~ t(n − 1). The extra uncertainty from estimating s gives heavier tails than the"
        " normal."
    ),
    use_cases=[
        "Small-sample inference (t-tests, confidence intervals), regression coefficients, robust modeling.",
    ],
    special_facts=[
        "Approaches Normal(0, 1) as ν → ∞; heavier tails for small ν.",
        "ν = 1 is the Cauchy distribution.",
        "If T ~ t(ν) then T² ~ F(1, ν).",
    ],
    aliases=("t", "students_t", "student"),
    alt_params={"df": same("nu")},
    mgf_domain=lambda P: (0.0, 0.0),
    xrange=lambda P, d: (-6.0, 6.0),
))

register(DistSpec(
    name="chi_squared",
    display="Chi-squared",
    kind="continuous",
    params=(_positive("k", "k", "degrees of freedom (number of squared normals)", 3.0),),
    scipy=lambda k: stats.chi2(k),
    parameter_space="k > 0 (an integer in the story)",
    support="x ∈ [0, ∞)  (x > 0 if k = 1)",
    density="f(x) = x^(k/2 − 1) e^(−x/2) / (2^(k/2) Γ(k/2))",
    cdf="F(x) = γ(k/2, x/2)/Γ(k/2) = P(k/2, x/2)",
    mean="k",
    variance="2k",
    mgf="M(t) = (1 − 2t)^(−k/2),  for t < ½",
    story=(
        "The sum of the squares of k independent standard normals, e.g. the squared distance from the origin of a"
        " random point in k dimensions."
    ),
    use_cases=[
        "Goodness-of-fit and independence tests, variance estimation, ANOVA, likelihood-ratio tests.",
    ],
    special_facts=[
        "Special case of Gamma: χ²(k) = Gamma(k/2, rate ½); mean = k, variance = 2k.",
        "χ²(2) = Exp(½).",
        "χ²(k₁) + χ²(k₂) = χ²(k₁ + k₂) for independent variables.",
        "(n − 1)s²/σ² ~ χ²(n − 1) for a normal sample.",
        "(U₁/k₁)/(U₂/k₂) ~ F(k₁, k₂) for independent chi-squared U₁, U₂.",
    ],
    aliases=("chi2", "chisquared", "chi_square"),
    alt_params={"df": same("k")},
    mgf_fn=lambda P, t: (1 - 2 * t) ** (-P["k"] / 2),
    mgf_domain=lambda P: (-INF, 0.5),
))

register(DistSpec(
    name="f",
    display="Snedecor F",
    kind="continuous",
    params=(_positive("d1", "d₁", "numerator degrees of freedom", 5.0),
            _positive("d2", "d₂", "denominator degrees of freedom", 10.0)),
    scipy=lambda d1, d2: stats.f(d1, d2),
    parameter_space="d₁ > 0,  d₂ > 0",
    support="x ∈ [0, ∞)",
    density="f(x) = √((d₁x)^d₁ d₂^d₂ / (d₁x + d₂)^(d₁ + d₂)) / (x B(d₁/2, d₂/2))",
    cdf="F(x) = I_{d₁x/(d₁x + d₂)}(d₁/2, d₂/2)",
    mean="d₂/(d₂ − 2) for d₂ > 2",
    variance="2d₂²(d₁ + d₂ − 2) / (d₁(d₂ − 2)²(d₂ − 4)) for d₂ > 4",
    mgf="Does not exist",
    story=(
        "The ratio of two independent chi-squared variables, each divided by its degrees of freedom: "
        "(U₁/d₁)/(U₂/d₂). E.g. the ratio of two sample variances from normal populations with equal variance."
    ),
    use_cases=[
        "ANOVA, comparing two variances, overall F-test in regression.",
    ],
    special_facts=[
        "If X ~ F(d₁, d₂) then 1/X ~ F(d₂, d₁).",
        "If T ~ t(ν) then T² ~ F(1, ν).",
        "d₁X/(d₁X + d₂) ~ Beta(d₁/2, d₂/2).",
        "As d₂ → ∞, d₁X → χ²(d₁).",
    ],
    aliases=("snedecor_f", "f_distribution", "fisher_f"),
    alt_params={"dfn": same("d1"), "dfd": same("d2")},
    mgf_domain=lambda P: (0.0, 0.0),
    xrange=lambda P, d: (0.0, min(d.ppf(0.99), 5.0)),
))

register(DistSpec(
    name="logistic",
    display="Logistic",
    kind="continuous",
    params=(_real("mu", "μ", "location (mean, median and mode)", 0.0), _positive("s", "s", "scale", 1.0)),
    scipy=lambda mu, s: stats.logistic(mu, s),
    parameter_space="μ ∈ ℝ,  s > 0",
    support="x ∈ ℝ",
    density="f(x) = e^(−(x − μ)/s) / (s (1 + e^(−(x − μ)/s))²)",
    cdf="F(x) = 1 / (1 + e^(−(x − μ)/s))  (the sigmoid)",
    mean="μ",
    variance="s²π²/3",
    mgf="M(t) = e^(μt) B(1 − st, 1 + st) = e^(μt) πst/sin(πst),  for |t| < 1/s",
    story=(
        "A symmetric bell curve whose CDF is the logistic (sigmoid) function. It is the noise distribution behind"
        " logistic regression and has heavier tails than the normal."
    ),
    use_cases=[
        "Latent-variable view of logistic regression, Elo ratings, growth curves.",
    ],
    special_facts=[
        "The CDF is the logistic (sigmoid) function.",
        "If U ~ Unif(0, 1), ln(U/(1 − U)) ~ Logistic(0, 1).",
        "The difference of two iid Gumbel(μ, β) is Logistic(0, β).",
    ],
    alt_params={"loc": same("mu"), "scale": same("s")},
    mgf_fn=lambda P, t: np.exp(P["mu"] * t) * special.beta(1 - P["s"] * t, 1 + P["s"] * t),
    mgf_domain=lambda P: (-1 / P["s"], 1 / P["s"]),
    xrange=lambda P, d: (P["mu"] - 6 * P["s"], P["mu"] + 6 * P["s"]),
))

register(DistSpec(
    name="gumbel",
    display="Gumbel",
    kind="continuous",
    params=(_real("mu", "μ", "location (mode)", 0.0), _positive("beta", "β", "scale", 1.0)),
    scipy=lambda mu, beta: stats.gumbel_r(mu, beta),
    parameter_space="μ ∈ ℝ,  β > 0",
    support="x ∈ ℝ",
    density="f(x) = (1/β) e^(−(z + e^(−z))),  z = (x − μ)/β",
    cdf="F(x) = exp(−e^(−(x − μ)/β))",
    mean="μ + βγ  (γ ≈ 0.5772 is the Euler–Mascheroni constant)",
    variance="π²β²/6",
    mgf="M(t) = Γ(1 − βt) e^(μt),  for t < 1/β",
    story=(
        "The maximum of many independent draws from a light-tailed distribution (normal, exponential, …), e.g. "
        "the highest flood level in a year."
    ),
    use_cases=[
        "Extreme values: floods, maximum wind speeds, earthquakes.",
        "The Gumbel-max trick for sampling from a categorical distribution.",
    ],
    special_facts=[
        "Type I extreme value distribution (for maxima; flip the sign for minima).",
        "If X ~ Exp(1) then μ − β ln X ~ Gumbel(μ, β).",
        "The difference of two iid Gumbels is Logistic.",
    ],
    aliases=("gumbel_r", "extreme_value", "gev_type1"),
    alt_params={"loc": same("mu"), "scale": same("beta")},
    mgf_fn=lambda P, t: special.gamma(1 - P["beta"] * t) * np.exp(P["mu"] * t),
    mgf_domain=lambda P: (-INF, 1 / P["beta"]),
    xrange=lambda P, d: (P["mu"] - 4 * P["beta"], P["mu"] + 8 * P["beta"]),
))

register(DistSpec(
    name="laplace",
    display="Laplace",
    kind="continuous",
    params=(_real("mu", "μ", "location (mean, median and mode)", 0.0), _positive("b", "b", "scale", 1.0)),
    scipy=lambda mu, b: stats.laplace(mu, b),
    parameter_space="μ ∈ ℝ,  b > 0",
    support="x ∈ ℝ",
    density="f(x) = 1/(2b) · e^(−|x − μ|/b)",
    cdf="F(x) = ½ e^((x − μ)/b) for x < μ;  1 − ½ e^(−(x − μ)/b) for x ≥ μ",
    mean="μ",
    variance="2b²",
    mgf="M(t) = e^(μt) / (1 − b²t²),  for |t| < 1/b",
    story=(
        "The difference of two independent exponential waiting times, centered at μ: a sharp peak with "
        "exponentially decaying tails on both sides (hence 'double exponential')."
    ),
    use_cases=[
        "L1 / LASSO regression (Laplace prior), differential privacy noise, robust error models.",
    ],
    special_facts=[
        "|X − μ| ~ Exp(1/b).",
        "The maximum likelihood estimate of μ is the sample median.",
    ],
    aliases=("double_exponential", "bilateral_exponential"),
    alt_params={"loc": same("mu"), "scale": same("b")},
    mgf_fn=lambda P, t: np.exp(P["mu"] * t) / (1 - P["b"] ** 2 * t ** 2),
    mgf_domain=lambda P: (-1 / P["b"], 1 / P["b"]),
    xrange=lambda P, d: (P["mu"] - 8 * P["b"], P["mu"] + 8 * P["b"]),
))

register(DistSpec(
    name="lognormal",
    display="Lognormal",
    kind="continuous",
    params=(_real("mu", "μ", "mean of ln X", 0.0), _positive("sigma", "σ", "standard deviation of ln X", 0.5)),
    scipy=lambda mu, sigma: stats.lognorm(sigma, scale=np.exp(mu)),
    parameter_space="μ ∈ ℝ,  σ > 0",
    support="x ∈ (0, ∞)",
    density="f(x) = 1/(xσ√(2π)) · exp(−(ln x − μ)²/(2σ²))",
    cdf="F(x) = Φ((ln x − μ)/σ)",
    mean="e^(μ + σ²/2)",
    variance="(e^(σ²) − 1) e^(2μ + σ²)",
    mgf="Does not exist for t > 0 (no closed form for t ≤ 0)",
    story=(
        "The product of many small, positive, independent factors, so that its logarithm is normal, e.g. anything"
        " that grows by compounding percentages."
    ),
    use_cases=[
        "Incomes, stock prices (geometric Brownian motion), particle sizes, gene expression, latency.",
    ],
    special_facts=[
        "ln X ~ Normal(μ, σ²); conversely e^Y is lognormal for Y normal.",
        "Products of independent lognormals are lognormal.",
        "Median = e^μ < mean = e^(μ + σ²/2).",
    ],
    aliases=("log_normal", "lognorm"),
    alt_params={"s": same("sigma")},
    mgf_domain=lambda P: (-INF, 0.0),
    xrange=lambda P, d: (0.0, float(np.exp(P["mu"] + 2 * P["sigma"]))),
))
