"""Utilities that work on several distributions at once."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

from ._plotting import COLOR_PALETTE, plot_distributions
from .distribution import Distribution, available

__all__ = ["plot", "available", "COLOR_PALETTE"]


def plot(
    dists: Sequence[Distribution],
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
    """Plot several (univariate) distributions on one set of axes, each in a different color.

    Parameters
    ----------
    dists
        The :class:`Distribution` objects. They can be of different families, and discrete and continuous
        distributions can be mixed (e.g. a Binomial and its Normal approximation).
    type
        ``"pmf"``/``"pdf"`` (aliases) or ``"cdf"``. A warning is logged only if *every* distribution
        mismatches the name (e.g. ``"pmf"`` for all-continuous distributions).
    out
        Path to save the figure to; the format comes from the extension.
    labels, colors
        One legend label / color per distribution. Defaults: the parameter values (with the family name
        when families differ) and a 9-color palette.
    xmin, xmax, ymin, ymax
        Axis limits. The x range defaults to the union of each distribution's natural plotting range.
    xlog, ylog
        Log-scale axes.
    samples
        Overlay sampled data as histograms (or empirical CDFs): an int draws that many samples from each
        distribution; otherwise a list with one array of data (or None) per distribution.
    y
        ``"fraction"`` (probabilities / densities, the default) or ``"count"`` (sample counts; the
        theoretical curves are scaled to expected counts). ``"count"`` needs ``samples``.
    bins
        Histogram bins for continuous samples (anything :func:`numpy.histogram_bin_edges` accepts).
    style
        For discrete distributions: ``"line"`` (markers joined by lines), ``"bar"`` or ``"stem"``.
        Defaults to each distribution's usual style (bars for Bernoulli, discrete uniform, categorical).
    plot_normal
        Overlay a dashed Normal curve: ``True`` uses each distribution's own mean and standard deviation;
        a ``(mu, sigma)`` tuple uses those values.
    title, xlabel, ylabel, legend_loc, figsize
        Usual matplotlib labeling options (sensible defaults are generated).
    dark
        Use matplotlib's ``dark_background`` style.
    ax
        Draw into an existing Axes instead of a new figure.
    show
        Call ``plt.show()``. Defaults to True unless ``ax`` is given.
    n_points
        Grid size for continuous curves.
    seed
        Random seed used when ``samples`` is an int.
    dpi
        Resolution for raster output files.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if isinstance(dists, Distribution):
        dists = [dists]
    return plot_distributions(
        dists, type=type, out=out, labels=labels, colors=colors, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
        xlog=xlog, ylog=ylog, samples=samples, y=y, bins=bins, style=style, plot_normal=plot_normal, title=title,
        xlabel=xlabel, ylabel=ylabel, legend_loc=legend_loc, figsize=figsize, dark=dark, ax=ax, show=show,
        n_points=n_points, seed=seed, dpi=dpi,
    )
