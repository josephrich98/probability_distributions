import matplotlib.pyplot as plt
import numpy as np
import pytest

from probability_distributions import Distribution, available, utils
from probability_distributions._interactive import _explore_matplotlib

UNIVARIATE = [n for n in available() if not Distribution(n).is_multivariate]


@pytest.mark.parametrize("name", UNIVARIATE)
@pytest.mark.parametrize("kind", ["density", "cdf"])
def test_plot_every_univariate(name, kind, tmp_path):
    d = Distribution(name)
    plot_type = "cdf" if kind == "cdf" else ("pmf" if d.is_discrete else "pdf")
    out = tmp_path / f"{name}_{plot_type}.png"
    ax = d.plot(plot_type, out=str(out), interactive=False, show=False)
    assert out.exists() and ax.get_title()


@pytest.mark.parametrize("name", ["multinomial", "multivariate_normal", "dirichlet"])
def test_plot_multivariate(name, tmp_path):
    out = tmp_path / f"{name}.png"
    Distribution(name).plot(out=str(out), interactive=False, show=False)
    assert out.exists()
    with pytest.raises(NotImplementedError):
        Distribution(name).plot("cdf", interactive=False, show=False)


def test_plot_options(tmp_path, caplog):
    d = Distribution("gamma", alpha=0.5, beta=1)
    ax = d.plot(interactive=False, show=False, xmin=0.01, xmax=10, xlog=True, ylog=True, dark=True)
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
    ax = d.plot(interactive=False, show=False, samples=1000, y="count", seed=0, plot_normal=True)
    assert ax.get_ylabel() == "Count"
    ax = Distribution("binomial", 30, 0.4).plot(interactive=False, show=False, samples=500, y="count",
                                                plot_normal=True, style="bar")
    assert len(ax.get_legend().get_texts()) == 3
    Distribution("normal").plot("cdf", interactive=False, show=False, samples=np.random.default_rng(0).normal(size=50))
    with pytest.raises(ValueError):
        d.plot(interactive=False, show=False, y="count")
    caplog.clear()
    d.plot("pmf", interactive=False, show=False)
    assert "continuous" in caplog.text


def test_density_cap_for_singular_pdf():
    ax = Distribution("beta", alpha=0.5, beta=0.5).plot(interactive=False, show=False)
    assert ax.get_ylim()[1] < 20


def test_utils_plot(tmp_path, caplog):
    dists = [Distribution("binomial", n=20, p=p) for p in (0.1, 0.5, 0.9)]
    ax = utils.plot(dists, out=str(tmp_path / "b.pdf"), show=False)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["n=20, p=0.1", "n=20, p=0.5", "n=20, p=0.9"]
    mixed = [Distribution("binomial", n=50, p=0.5), Distribution("normal", mu=25, sigma=np.sqrt(12.5))]
    caplog.clear()
    ax = utils.plot(mixed, "pdf", show=False, labels=["Binomial", "Normal approx."], colors=["C0", "C1"])
    assert caplog.text == ""  # mixed families: no pmf/pdf warning
    utils.plot([Distribution("bernoulli", p=0.2), Distribution("bernoulli", p=0.7)], show=False)
    utils.plot([Distribution("exponential", lam=l) for l in (0.5, 1, 2)], "cdf", show=False, samples=200, seed=1)
    with pytest.raises(ValueError):
        utils.plot([Distribution("dirichlet")], show=False)


def test_matplotlib_interactive_updates_params():
    d = Distribution("binomial", n=10, p=0.3)
    ax = _explore_matplotlib(d, None, {"n": (1, 40)}, {})
    widgets = ax.figure._probability_distributions_widgets
    widgets["p"].set_val(0.7)
    widgets["n"].set_val(25)
    assert d.params == {"n": 25, "p": pytest.approx(0.7)}
    assert "n=25" in ax.get_title()

    normal = Distribution("normal")
    ax = _explore_matplotlib(normal, "cdf", {}, {})
    boxes = ax.figure._probability_distributions_widgets
    boxes["sigma"].set_val("-1")  # invalid -> error shown, params unchanged
    assert "⚠" in ax.get_title() and normal.sigma == 1.0
    boxes["sigma"].set_val("2")
    assert normal.sigma == 2.0

    mvn = Distribution("multivariate_normal")
    ax = _explore_matplotlib(mvn, None, {}, {})
    ax.figure._probability_distributions_widgets["cov"].set_val("1, 0.9; 0.9, 1")
    assert mvn.params["cov"][0, 1] == 0.9
    plt.close("all")


def test_ipywidgets_interactive_updates_params(monkeypatch):
    pytest.importorskip("ipywidgets")
    import ipywidgets as W

    from probability_distributions import _interactive

    shown = []
    monkeypatch.setattr(_interactive, "_in_notebook", lambda: True)
    monkeypatch.setattr("IPython.display.display", lambda obj: shown.append(obj))
    d = Distribution("beta_binomial", n=10, alpha=2, beta=2)
    assert d.plot(sliders={"alpha": (0.1, 10)}) is None  # auto-interactive in a "notebook"
    box = shown[0]
    controls = {c.description: c for c in box.children if hasattr(c, "description") and c.description}
    assert isinstance(controls["n"], W.IntText)  # unbounded -> box
    assert isinstance(controls["α"], W.FloatSlider)  # forced slider
    controls["α"].value = 5.0
    controls["n"].value = 20
    assert d.params == {"n": 20, "alpha": 5.0, "beta": 2.0}
    controls["n"].value = -3  # invalid -> error message, params unchanged
    assert "⚠" in box.children[3].value and d.n == 20

    shown.clear()
    p = Distribution("bernoulli", p=0.2)
    p.plot()
    slider = shown[0].children[0]
    assert isinstance(slider, W.FloatSlider) and (slider.min, slider.max) == (0.0, 1.0)
