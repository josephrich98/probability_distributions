import math

import numpy as np
import pytest

from probability_distributions import Distribution, available
from probability_distributions._interactive import from_text, slider_range, to_text

UNIVARIATE = [n for n in available() if not Distribution(n).is_multivariate]
MULTIVARIATE = available("multivariate")


@pytest.mark.parametrize("name", available())
def test_defaults_and_text(name):
    d = Distribution(name)
    for text in (d.cdf(), d.mgf(), d.parameter_space(), d.support(), d.story(), d.expected_value(formula=True),
                 d.variance(formula=True)):
        assert isinstance(text, str) and text
    assert d.use_cases() and d.special_facts()
    assert d._spec.density_name in d.describe()


@pytest.mark.parametrize("name", UNIVARIATE)
def test_cdf_and_density(name):
    d = Distribution(name)
    lo, hi = d.support_bounds()
    x = np.linspace(d.scipy.ppf(0.01), d.scipy.ppf(0.99), 50)
    F = d.cdf(x)
    assert np.all(np.diff(F) >= -1e-12) and np.all((F >= 0) & (F <= 1))
    if d.is_discrete:
        xs = np.arange(lo, min(hi, d.scipy.ppf(1 - 1e-12)) + 1)
        assert d._density(xs).sum() == pytest.approx(1, abs=1e-6)


@pytest.mark.parametrize("name", UNIVARIATE)
def test_mgf_matches_definition_and_mean(name):
    d = Distribution(name)
    lo, hi = d._spec.mgf_domain(d._params)
    t = min(0.05, hi / 4) if hi > 0 else max(-0.05, lo / 4)
    if lo < t < hi and t != 0:
        expected = d.scipy.expect(lambda x: np.exp(t * x))
        assert d.mgf(t) == pytest.approx(expected, rel=1e-5)
    # M'(0) = E[X] whenever the MGF exists around 0
    if lo < 0 < hi:
        h = 1e-5
        assert (d.mgf(h) - d.mgf(-h)) / (2 * h) == pytest.approx(d.expected_value(), rel=1e-4, abs=1e-6)
    assert d.mgf(0.0) == 1.0


def test_mgf_outside_domain_warns(caplog):
    assert math.isnan(Distribution("exponential", lam=1).mgf(2.0))
    assert "does not exist" in caplog.text
    assert np.isnan(Distribution("cauchy").mgf(np.array([0.1, -0.1]))).all()


def test_multivariate_mgf():
    d = Distribution("multivariate_normal", mean=[1, 2], cov=[[1, 0.3], [0.3, 2]])
    t = np.array([0.1, -0.2])
    assert d.mgf(t) == pytest.approx(np.exp(t @ [1, 2] + 0.5 * t @ d.params["cov"] @ t))
    m = Distribution("multinomial", n=4, p=[0.2, 0.3, 0.5])
    assert m.mgf(np.zeros(3)) == pytest.approx(1)


@pytest.mark.parametrize("name", available())
def test_sampling(name, tmp_path):
    d = Distribution(name)
    one = d.sample(seed=1)
    assert np.ndim(one) == (1 if d.is_multivariate else 0)
    many = d.sample(100, seed=1)
    assert many.shape[0] == 100
    np.testing.assert_array_equal(many, d.sample(100, seed=1))
    out = tmp_path / "sub" / "samples.txt"
    d.sample(10, out=str(out), seed=2)
    loaded = np.loadtxt(out)
    assert loaded.shape[0] == 10


def test_pmf_pdf_alias_warnings(caplog):
    normal = Distribution("normal")
    assert normal.pmf(0) == pytest.approx(1 / math.sqrt(2 * math.pi))
    assert "continuous" in caplog.text
    caplog.clear()
    binom = Distribution("binomial", 20, 0.3)
    assert binom.pdf(6) == pytest.approx(binom.pmf(6))
    assert "discrete" in caplog.text
    caplog.clear()
    binom.pmf(3)
    normal.pdf(0)
    assert caplog.text == ""


def test_parameters_and_validation():
    d = Distribution("binomial", 20, p=0.3)
    assert d.n == 20 and d.params == {"n": 20, "p": 0.3}
    assert d.expected_value() == pytest.approx(6) and d.variance() == pytest.approx(4.2)
    d.set_params(p=0.5)
    assert d.expected_value() == pytest.approx(10)
    with pytest.raises(ValueError):
        d.set_params(p=1.5)
    assert d.p == 0.5  # unchanged after a failed update
    with pytest.raises(ValueError):
        Distribution("binomial", n=2.5)
    with pytest.raises(TypeError):
        Distribution("binomial", q=0.1)
    with pytest.raises(ValueError):
        Distribution("uniform", a=2, b=1)
    with pytest.raises(ValueError):
        Distribution("hypergeometric", N=10, K=11, n=2)
    with pytest.raises(ValueError):
        Distribution("not_a_distribution")
    with pytest.raises(ValueError):
        Distribution("multivariate_normal", mean=[0, 0], cov=[[1, 2], [2, 1]])


def test_aliases_and_presets():
    assert Distribution("standard_normal").params == {"mu": 0.0, "sigma": 1.0}
    assert Distribution("Chi-Squared", 4).name == "chi_squared"
    assert Distribution("double exponential").name == "laplace"
    assert Distribution("t", 5).name == "student_t"


def test_alternative_parameterizations():
    g = Distribution("gamma", k=2, theta=3)
    assert g.params == {"alpha": 2.0, "beta": pytest.approx(1 / 3)}
    assert g.expected_value() == pytest.approx(6)  # kθ
    assert Distribution("gamma", shape=2, rate=4).expected_value() == pytest.approx(0.5)
    assert Distribution("exponential", scale=2).lam == pytest.approx(0.5)
    assert Distribution("normal", variance=4).sigma == pytest.approx(2)
    assert Distribution("student_t", df=7).nu == 7
    assert Distribution("multivariate_normal", mean=[1, 2], cov=np.eye(2)).params["mu"].tolist() == [1, 2]
    assert g.copy(theta=1).beta == pytest.approx(1)
    g.set_params(scale=0.5)
    assert g.beta == pytest.approx(2)
    with pytest.raises(TypeError):
        Distribution("gamma", beta=1, theta=2)
    with pytest.raises(ValueError):
        Distribution("gamma", theta=0)
    with pytest.raises(ValueError):
        Distribution("gamma", theta=-1)
    assert "Also accepts" in Distribution("gamma").describe()


def test_categorical_fills_remaining_probability():
    d = Distribution("categorical", p=[0.5, 0.3])
    np.testing.assert_allclose(d.params["p"], [0.5, 0.3, 0.2])
    assert d.pmf(3) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        Distribution("categorical", p=[0.7, 0.6])


def test_known_values():
    assert Distribution("poisson", lam=3).variance() == pytest.approx(3)
    assert Distribution("gamma", alpha=3, beta=2).expected_value() == pytest.approx(1.5)
    assert Distribution("exponential", lam=2).cdf(1) == pytest.approx(1 - math.exp(-2))
    assert Distribution("geometric", 0.25).expected_value() == pytest.approx(4)
    assert Distribution("geometric_failures", 0.25).expected_value() == pytest.approx(3)
    assert math.isnan(Distribution("cauchy").expected_value())
    assert Distribution("pareto", alpha=1, xm=1).expected_value() == math.inf
    np.testing.assert_allclose(Distribution("dirichlet", alpha=[1, 1, 2]).expected_value(), [0.25, 0.25, 0.5])
    np.testing.assert_allclose(Distribution("dirichlet", alpha=[1, 1, 2]).variance(),
                               np.cov(Distribution("dirichlet", alpha=[1, 1, 2]).sample(200000, seed=0).T),
                               atol=2e-3)
    assert Distribution("dirichlet", alpha=[2, 2, 2]).pdf([0.2, 0.3, 0.5]) > 0
    assert Distribution("discrete_uniform", 1, 6).support().endswith("(here: {1, 2, …, 6})")
    assert "here: {0, 1}" in Distribution("bernoulli").support()


def test_control_specs():
    p = Distribution("binomial")._spec.params
    assert slider_range(p[1], 0.3, None) == (0.0, 1.0, 0.01)  # p ∈ [0, 1] -> slider
    assert slider_range(p[0], 10, None) is None  # n unbounded -> box
    assert slider_range(p[0], 10, (1, 50)) == (1.0, 50.0, 1.0)
    geometric_p = Distribution("geometric")._spec.params[0]
    assert slider_range(geometric_p, 0.5, None)[0] == pytest.approx(0.01)  # open bound nudged
    cov = Distribution("multivariate_normal")._spec.params[1]
    assert from_text(cov, to_text(np.array([[1.0, 0.5], [0.5, 2.0]]))) == [[1.0, 0.5], [0.5, 2.0]]
