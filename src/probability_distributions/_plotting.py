"""Matplotlib drawing shared by :meth:`Distribution.plot` and :func:`utils.plot`."""

from __future__ import annotations

import contextlib
import logging
import math
import os
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy import stats


if TYPE_CHECKING:  # pragma: no cover
    from .distribution import Distribution

logger = logging.getLogger("probability_distributions")

COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#bcbd22", "#17becf",
]
NORMAL_COLOR = "gray"


# ---------------------------------------------------------------------- helpers
def style_context(dark: bool):
    return plt.style.context("dark_background") if dark else contextlib.nullcontext()


def resolve_type(type: Optional[str], dists: Sequence["Distribution"]) -> str:
    """Map ``"pmf"``/``"pdf"``/``"cdf"`` to ``"density"`` or ``"cdf"``, warning on a pmf/pdf mismatch."""
    if type is None:
        return "density"
    key = type.lower()
    if key == "cdf":
        return "cdf"
    if key not in ("pmf", "pdf", "density"):
        raise ValueError(f"type must be 'pmf', 'pdf' or 'cdf', got {type!r}")
    if key == "pmf" and all(d.is_continuous for d in dists):
        logger.warning("type='pmf' requested for a continuous distribution; plotting the PDF.")
    elif key == "pdf" and all(d.is_discrete for d in dists):
        logger.warning("type='pdf' requested for a discrete distribution; plotting the PMF.")
    return "density"


def default_xrange(dist: "Distribution") -> Tuple[float, float]:
    """A sensible plotting range covering the bulk of the distribution."""
    spec, frozen, P = dist._spec, dist.scipy, dist._params
    if spec.xrange is not None:
        lo, hi = spec.xrange(P, frozen)
        return float(lo), float(hi)
    slo, shi = (float(v) for v in frozen.support())
    if dist.is_discrete:
        if math.isfinite(shi) and shi - slo <= 200:
            return slo, shi
        lo = max(slo, float(frozen.ppf(0.0005)))
        hi = max(float(frozen.ppf(0.9995)), lo + 5)
        return lo, min(hi, shi)
    hi = shi if math.isfinite(shi) else float(frozen.ppf(0.999))
    if math.isfinite(slo):
        q = float(frozen.ppf(0.001))
        lo = slo if q - slo < 0.25 * (hi - slo) else q
    else:
        lo = float(frozen.ppf(0.001))
    return lo, hi


def _normal_params(dist: "Distribution", plot_normal: Any) -> Optional[Tuple[float, float]]:
    if plot_normal is True:
        mu, var = dist.expected_value(), dist.variance()
        if not (math.isfinite(mu) and math.isfinite(var)) or var <= 0:
            logger.warning("%s has no finite mean/variance; skipping the Normal overlay.", dist)
            return None
        return mu, math.sqrt(var)
    mu, sigma = plot_normal
    return float(mu), float(sigma)


def _parse_y(y: str) -> bool:
    """True for counts, False for fractions/probabilities."""
    key = y.lower()
    if key in ("count", "counts"):
        return True
    if key in ("fraction", "fractions", "probability", "proportion", "density"):
        return False
    raise ValueError(f"y must be 'fraction' or 'count', got {y!r}")


def _resolve_samples(samples: Any, dists: Sequence["Distribution"], seed: Any) -> List[Optional[np.ndarray]]:
    if samples is None:
        return [None] * len(dists)
    if np.isscalar(samples):
        rng = np.random.default_rng(seed)
        return [np.asarray(d.scipy.rvs(size=int(samples), random_state=rng)).ravel() for d in dists]
    samples = list(samples)
    if len(samples) != len(dists):
        raise ValueError(f"got {len(samples)} sample arrays for {len(dists)} distributions")
    return [None if s is None else np.asarray(s, dtype=float).ravel() for s in samples]


def save_and_show(fig, out: Optional[str], show: bool, dpi: Any = None) -> None:
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        fig.savefig(out, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    elif out:
        plt.close(fig)


# ---------------------------------------------------------------------- univariate drawing
def _draw_discrete(ax, dist, kind, lo, hi, *, color, label, style, sample, counts, bar_slot):
    """Draw one discrete distribution; returns the factor its theoretical curve was scaled by."""
    frozen = dist.scipy
    x = np.arange(math.ceil(lo), math.floor(hi) + 1)
    n_samples = len(sample) if sample is not None else 1
    scale = n_samples if counts else 1.0
    offset, width = bar_slot

    if kind == "density":
        y = frozen.pmf(x) * scale
        if sample is not None:
            values, freq = np.unique(np.round(sample).astype(int), return_counts=True)
            keep = (values >= x[0]) & (values <= x[-1])
            emp = freq[keep] if counts else freq[keep] / len(sample)
            ax.bar(values[keep] + offset, emp, width=width, color=color, alpha=0.35, label=f"{label} (samples)")
            ax.plot(x, y, marker="o", linestyle="-" if style == "line" else "none", color=color, ms=5, lw=1,
                    label=label)
        elif style == "bar":
            ax.bar(x + offset, y, width=width, color=color, alpha=0.8, label=label, align="center")
        elif style == "stem":
            ax.vlines(x, 0, y, color=color, lw=1.5)
            ax.plot(x, y, "o", color=color, ms=5, label=label)
        elif style == "line":
            ax.plot(x, y, marker="o", linestyle="-", color=color, ms=5, lw=1, label=label)
        else:
            raise ValueError(f"style must be 'line', 'bar' or 'stem', got {style!r}")
    else:
        x_ext = np.concatenate([[x[0] - 1], x, [x[-1] + 1]])
        y = frozen.cdf(x_ext) * scale
        ax.step(x_ext, y, where="post", color=color, lw=1.5, label=label)
        ax.plot(x, frozen.cdf(x) * scale, "o", color=color, ms=4)
        if sample is not None:
            _draw_ecdf(ax, sample, counts, color, label)
    return scale


def _draw_ecdf(ax, sample, counts, color, label):
    xs = np.sort(sample)
    ys = np.arange(1, len(xs) + 1) / (1 if counts else len(xs))
    ax.step(xs, ys, where="post", color=color, alpha=0.45, lw=3, label=f"{label} (samples)")


def _draw_continuous(ax, dist, kind, lo, hi, *, color, label, sample, counts, bins, n_points, xlog):
    """Draw one continuous distribution; returns (y cap or None, factor the curve was scaled by or None)."""
    frozen = dist.scipy
    log_grid = xlog and lo > 0
    x = np.geomspace(lo, hi, n_points) if log_grid else np.linspace(lo, hi, n_points)
    n_samples = len(sample) if sample is not None else 1
    scale = 1.0
    stepwise_expected = None
    cap = None

    if sample is not None and kind == "density":
        inside = sample[(sample >= lo) & (sample <= hi)]
        if log_grid:
            n_bins = bins if isinstance(bins, (int, np.integer)) else len(np.histogram_bin_edges(np.log(inside), bins))
            edges = np.geomspace(lo, hi, n_bins + 1)
        else:
            edges = np.histogram_bin_edges(inside, bins=bins, range=(lo, hi))
        freq, edges = np.histogram(inside, bins=edges)
        widths = np.diff(edges)
        heights = freq if counts else freq / (n_samples * widths)
        ax.bar(edges[:-1], heights, width=widths, align="edge", color=color, alpha=0.35, label=f"{label} (samples)")
        if counts:
            if log_grid:  # unequal bins: compare against the expected count in each bin
                stepwise_expected = (edges, n_samples * np.diff(frozen.cdf(edges)))
            else:
                scale = n_samples * widths[0]
    elif sample is not None and kind == "cdf":
        _draw_ecdf(ax, sample, counts, color, label)
        scale = n_samples if counts else 1.0

    if kind == "density":
        with np.errstate(divide="ignore"):
            y = frozen.pdf(x)
        finite = y[np.isfinite(y)]
        # Cap the y axis when the density diverges at an edge of the range (e.g. Gamma with α < 1)
        if finite.size and np.argmax(np.where(np.isfinite(y), y, np.inf)) in (0, len(y) - 1):
            q90 = np.quantile(finite, 0.9)
            if np.max(finite) > 3 * q90 > 0:
                cap = 3 * q90 * scale
    else:
        y = frozen.cdf(x)

    if stepwise_expected is not None:
        edges, expected = stepwise_expected
        ax.stairs(expected, edges, color=color, lw=2, label=label)
        return cap, None
    ax.plot(x, y * scale, color=color, lw=2, label=label)
    return cap, scale


def _draw_normal(ax, params, kind, lo, hi, scale, n_points, xlog, color):
    log_grid = xlog and lo > 0
    x = np.geomspace(lo, hi, n_points) if log_grid else np.linspace(lo, hi, n_points)
    normal = stats.norm(*params)
    y = normal.pdf(x) if kind == "density" else normal.cdf(x)
    ax.plot(x, y * scale, color=color, ls="--", lw=1.5,
            label=f"Normal({params[0]:.3g}, {params[1]:.3g}²)")


# ---------------------------------------------------------------------- main entry point
def plot_distributions(
    dists: Sequence["Distribution"],
    type: Optional[str] = None,
    out: Optional[str] = None,
    *,
    labels: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    xlog: bool = False,
    ylog: bool = False,
    samples: Any = None,
    y: str = "fraction",
    bins: Any = "auto",
    style: Optional[str] = None,
    plot_normal: Any = False,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend_loc: str = "best",
    figsize: Tuple[float, float] = (8, 5),
    dark: bool = False,
    ax: Any = None,
    show: Optional[bool] = None,
    n_points: int = 400,
    seed: Any = None,
    dpi: Any = None,
):
    dists = list(dists)
    if not dists:
        raise ValueError("no distributions to plot")
    multivariate = [d for d in dists if d.is_multivariate]
    if multivariate:
        raise ValueError(f"{multivariate[0]} is multivariate; plot it on its own with its .plot() method")
    kind = resolve_type(type, dists)
    counts = _parse_y(y)
    sample_list = _resolve_samples(samples, dists, seed)
    if counts and any(s is None for s in sample_list):
        raise ValueError("y='count' needs samples (an int number of samples to draw, or the data itself)")
    if show is None:
        show = ax is None
    if labels is not None and len(labels) != len(dists):
        raise ValueError(f"got {len(labels)} labels for {len(dists)} distributions")
    if colors is not None and len(colors) != len(dists):
        raise ValueError(f"got {len(colors)} colors for {len(dists)} distributions")

    same_family = len({d.name for d in dists}) == 1
    if labels is None:
        labels = [d.param_string() if same_family else str(d) for d in dists]
    if colors is None:
        colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(dists))]

    # x range: union of each distribution's default range (and its samples), overridden by xmin/xmax
    ranges = [default_xrange(d) for d in dists]
    lo = min(r[0] for r in ranges)
    hi = max(r[1] for r in ranges)
    for d, s in zip(dists, sample_list):
        if s is not None and d.is_discrete and s.size:
            lo, hi = min(lo, float(s.min())), max(hi, float(s.max()))
    lo = lo if xmin is None else xmin
    hi = hi if xmax is None else xmax
    if xlog and lo <= 0:
        lo = hi * 1e-4 if xmin is None else lo
        if lo <= 0:
            raise ValueError("xlog=True needs xmin > 0")
    if hi <= lo:
        raise ValueError(f"empty x range [{lo:g}, {hi:g}]")

    bar_styles = [(style or d._spec.style) == "bar" or s is not None
                  for d, s in zip(dists, sample_list) if d.is_discrete]
    n_bars = max(1, sum(bar_styles))
    width = 0.6 if n_bars == 1 else 0.8 / n_bars

    with style_context(dark):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        caps = []
        scales = []
        bar_index = 0
        for d, label, color, sample in zip(dists, labels, colors, sample_list):
            if d.is_discrete:
                d_style = style or d._spec.style
                uses_bar = d_style == "bar" or sample is not None
                offset = (bar_index - (n_bars - 1) / 2) * width if uses_bar else 0.0
                bar_index += uses_bar
                d_lo, d_hi = max(lo, d.support_bounds()[0]), min(hi, d.support_bounds()[1])
                if math.floor(d_hi) < math.ceil(d_lo):
                    logger.warning("%s has no support inside x ∈ [%g, %g]; skipping it.", d, lo, hi)
                    continue
                scales.append(_draw_discrete(ax, d, kind, d_lo, d_hi, color=color, label=label, style=d_style,
                                             sample=sample, counts=counts, bar_slot=(offset, width)))
            else:
                cap, scale = _draw_continuous(ax, d, kind, lo, hi, color=color, label=label, sample=sample,
                                              counts=counts, bins=bins, n_points=n_points, xlog=xlog)
                caps.append(cap)
                scales.append(scale)

        # Normal overlay: True = each distribution's own mean/sd, (mu, sigma) = one fixed Normal
        if plot_normal is True:
            for d, color, scale in zip(dists, colors, scales):
                params = _normal_params(d, True)
                if params is not None and scale is not None:
                    pad = 0.5 if d.is_discrete else 0.0
                    _draw_normal(ax, params, kind, lo - pad, hi + pad, scale, n_points, xlog,
                                 NORMAL_COLOR if len(dists) == 1 else color)
        elif plot_normal is not False:
            scale = next((s for s in scales if s is not None), 1.0)
            _draw_normal(ax, _normal_params(dists[0], plot_normal), kind, lo, hi, scale, n_points, xlog,
                         NORMAL_COLOR)

        _decorate(ax, dists, kind, counts, labels, same_family, caps, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                  xlog=xlog, ylog=ylog, title=title, xlabel=xlabel, ylabel=ylabel, legend_loc=legend_loc)
        if ax.figure is fig and len(fig.axes) == 1:
            fig.tight_layout()
    save_and_show(fig, out, show, dpi)
    return ax


def _decorate(ax, dists, kind, counts, labels, same_family, caps, *, xmin, xmax, ymin, ymax, xlog, ylog, title,
              xlabel, ylabel, legend_loc):
    all_discrete = all(d.is_discrete for d in dists)
    all_continuous = all(d.is_continuous for d in dists)
    if kind == "density":
        what = "PMF" if all_discrete else "PDF" if all_continuous else "PMF / PDF"
    else:
        what = "CDF"

    if title is None:
        if len(dists) == 1:
            title = f"{dists[0].display_name} {what}: {labels[0]}"
        elif same_family:
            title = f"{dists[0].display_name} {what}"
        else:
            title = what
    if ylabel is None:
        if kind == "cdf":
            ylabel = "Cumulative count" if counts else "F(x) = P(X ≤ x)"
        elif counts:
            ylabel = "Count"
        else:
            ylabel = "P(X = x)" if all_discrete else "f(x)" if all_continuous else "P(X = x) / f(x)"
    ax.set_title(title)
    ax.set_xlabel(xlabel if xlabel is not None else "x")
    ax.set_ylabel(ylabel)

    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    if all_discrete and not xlog:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if len(dists) == 1 and dists[0].name == "categorical":
            k = len(dists[0]._params["p"])
            ax.set_xticks(np.arange(1, k + 1))
            ax.set_xticklabels([f"$x_{{{i}}}$" for i in range(1, k + 1)])

    if xmin is not None or xmax is not None:
        ax.set_xlim(left=xmin, right=xmax)
    caps = [c for c in caps if c is not None]
    top = ymax
    if top is None and caps and len(caps) == len([d for d in dists if d.is_continuous]):
        top = max(caps)
    if not ylog:
        ax.set_ylim(bottom=0 if ymin is None else ymin, top=top)
    elif ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)

    handles, handle_labels = ax.get_legend_handles_labels()
    if len(handles) > 1:
        ax.legend(frameon=False, loc=legend_loc)


# ---------------------------------------------------------------------- multivariate (3-D) drawing
def plot_multivariate(
    dist: "Distribution",
    type: Optional[str] = None,
    out: Optional[str] = None,
    *,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 5),
    dark: bool = False,
    ax: Any = None,
    show: Optional[bool] = None,
    cmap: str = "viridis",
    n_points: int = 150,
    dpi: Any = None,
    **ignored: Any,
):
    """3-D plots: Multinomial with k = 3 (bars over x₁, x₂), Dirichlet with k = 3 (surface over the simplex),
    Multivariate Normal with k = 2 (surface)."""
    if resolve_type(type, [dist]) == "cdf":
        raise NotImplementedError(f"CDF plots are not supported for the {dist.display_name} distribution")
    unused = {k for k, v in ignored.items() if v not in (None, False)}
    if unused:
        logger.warning("Ignoring options not supported by 3-D plots: %s", ", ".join(sorted(unused)))
    if show is None:
        show = ax is None
    P = dist._params

    with style_context(dark):
        created = ax is None
        if created:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure
            if getattr(ax, "name", "") != "3d":
                raise ValueError("ax must be a 3-D axes (projection='3d')")

        if dist.name == "multinomial":
            n, p = P["n"], P["p"]
            if len(p) != 3:
                raise NotImplementedError("Multinomial plots need exactly k = 3 categories "
                                          "(k = 2 is the Binomial: Distribution('binomial', ...))")
            x1, x2 = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
            mask = x1 + x2 <= n
            x1, x2 = x1[mask], x2[mask]
            z = dist.pmf(np.stack([x1, x2, n - x1 - x2], axis=1))
            norm = plt.Normalize(vmin=z.min(), vmax=z.max())
            ax.bar3d(x1 - 0.4, x2 - 0.4, np.zeros_like(z), 0.8, 0.8, z, color=plt.get_cmap(cmap)(norm(z)),
                     alpha=0.9, shade=True)
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.set_zlabel("$P(X_1=x_1, X_2=x_2)$")
            cbar_label = "P(X₁, X₂)"
        elif dist.name == "dirichlet":
            alpha = P["alpha"]
            if len(alpha) != 3:
                raise NotImplementedError("Dirichlet plots need exactly k = 3 components "
                                          "(k = 2 is the Beta: Distribution('beta', ...))")
            g = np.linspace(0.005, 0.995, n_points)
            x1, x2 = np.meshgrid(g, g)
            mask = x1 + x2 < 0.995
            x1, x2 = x1[mask], x2[mask]
            z = dist.pdf(np.stack([x1, x2, 1 - x1 - x2], axis=1))
            finite = np.isfinite(z)
            z_cap = np.quantile(z[finite], 0.99)
            z = np.minimum(np.where(finite, z, z_cap), z_cap)
            norm = plt.Normalize(vmin=z.min(), vmax=z.max())
            ax.plot_trisurf(x1, x2, z, cmap=cmap, norm=norm, linewidth=0.05, antialiased=True, alpha=0.95)
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.set_zlabel("$f(x_1, x_2)$")
            cbar_label = "density"
        elif dist.name == "multivariate_normal":
            mean, cov = P["mu"], P["cov"]
            if len(mean) != 2:
                raise NotImplementedError("Multivariate Normal plots need exactly 2 dimensions")
            sd = np.sqrt(np.diag(cov))
            gx = np.linspace(mean[0] - 4 * sd[0], mean[0] + 4 * sd[0], n_points)
            gy = np.linspace(mean[1] - 4 * sd[1], mean[1] + 4 * sd[1], n_points)
            X, Y = np.meshgrid(gx, gy)
            z = dist.pdf(np.dstack((X, Y)))
            norm = plt.Normalize(vmin=z.min(), vmax=z.max())
            ax.plot_surface(X, Y, z, cmap=cmap, norm=norm, linewidth=0, antialiased=True, alpha=0.9)
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.set_zlabel("$f(x, y)$")
            cbar_label = "density"
        else:  # pragma: no cover
            raise NotImplementedError(dist.name)

        ax.set_title(title if title is not None else f"{dist.display_name}: {dist.param_string()}", pad=10)
        if created:
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array(z)
            fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.15, label=cbar_label)
    save_and_show(fig, out, show, dpi)
    return ax
