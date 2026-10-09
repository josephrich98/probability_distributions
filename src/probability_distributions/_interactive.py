"""Interactive parameter controls: ipywidgets in Jupyter, matplotlib widgets in a GUI window.

Each parameter gets a slider if it lives in a bounded range (e.g. p ∈ [0, 1]) or if the caller passes a
``sliders={"name": (min, max)}`` range for it, and a text box otherwise.
"""

from __future__ import annotations

import html
import math
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import matplotlib
import numpy as np

from ._spec import Param

if TYPE_CHECKING:  # pragma: no cover
    from .distribution import Distribution

_NON_INTERACTIVE_BACKENDS = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}


# ---------------------------------------------------------------------- environment detection
def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and "IPKernelApp" in getattr(shell, "config", {})


def _has_ipywidgets() -> bool:
    try:
        import ipywidgets  # noqa: F401
    except ImportError:
        return False
    return True


def mode() -> Optional[str]:
    """``"ipywidgets"``, ``"matplotlib"`` or ``None`` (no interactive display available)."""
    if _in_notebook():
        return "ipywidgets" if _has_ipywidgets() else None
    backend = matplotlib.get_backend().lower()
    if backend in _NON_INTERACTIVE_BACKENDS or "inline" in backend:
        return None
    return "matplotlib"


# ---------------------------------------------------------------------- control specs
def slider_range(param: Param, value: Any, override: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float, float]]:
    """(min, max, step) if this parameter should be a slider, else None."""
    if param.kind not in ("int", "float"):
        return None
    if override is not None:
        lo, hi = (float(v) for v in override)
    elif param.bounded:
        lo, hi = param.lower, param.upper
    else:
        return None
    if param.kind == "int":
        return float(math.ceil(lo)), float(math.floor(hi)), 1.0
    step = 10 ** math.floor(math.log10((hi - lo) / 100)) if hi > lo else 0.01
    if override is None:  # keep open bounds out of reach
        lo = lo if param.lower_inclusive else lo + step
        hi = hi if param.upper_inclusive else hi - step
    return min(lo, value), max(hi, value), step


def to_text(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.ndim == 2:
            return "; ".join(", ".join(f"{v:g}" for v in row) for row in value)
        return ", ".join(f"{v:g}" for v in value)
    return f"{value:g}" if isinstance(value, float) else str(value)


def from_text(param: Param, text: str) -> Any:
    text = text.strip().strip("[]()")
    try:
        if param.kind == "matrix":
            return [[float(v) for v in row.replace("[", "").replace("]", "").split(",") if v.strip()]
                    for row in text.split(";")]
        if param.kind == "vector":
            return [float(v) for v in text.replace(";", ",").split(",") if v.strip()]
        return float(text)
    except ValueError:
        raise ValueError(f"could not parse {param.name} from {text!r}") from None


# ---------------------------------------------------------------------- entry point
def explore(dist: "Distribution", type: Optional[str] = None, sliders: Optional[Dict[str, Tuple[float, float]]] = None,
            **plot_kwargs: Any) -> Any:
    unknown = set(sliders or {}) - set(dist.param_names)
    if unknown:
        raise ValueError(f"sliders given for unknown parameter(s) {sorted(unknown)}")
    plot_kwargs.pop("show", None)
    plot_kwargs.pop("out", None)
    if mode() == "ipywidgets":
        return _explore_ipywidgets(dist, type, sliders or {}, plot_kwargs)
    return _explore_matplotlib(dist, type, sliders or {}, plot_kwargs)


def _explore_ipywidgets(dist, type, sliders, plot_kwargs):
    import ipywidgets as W
    import matplotlib.pyplot as plt
    from IPython.display import display

    controls = {}
    for p in dist._spec.params:
        value = dist._params[p.name]
        rng = slider_range(p, value, sliders.get(p.name))
        common = dict(description=p.symbol, tooltip=p.description,
                      style={"description_width": "initial"}, layout=W.Layout(width="420px"))
        if rng is not None:
            lo, hi, step = rng
            if p.kind == "int":
                w = W.IntSlider(value=value, min=int(lo), max=int(hi), step=1, continuous_update=True, **common)
            else:
                decimals = max(0, -int(math.floor(math.log10(step))))
                w = W.FloatSlider(value=value, min=lo, max=hi, step=step, readout_format=f".{decimals}f",
                                  continuous_update=True, **common)
        elif p.kind == "int":
            w = W.IntText(value=value, **common)
        elif p.kind == "float":
            w = W.FloatText(value=value, **common)
        else:
            w = W.Text(value=to_text(value), continuous_update=False, **common)
        controls[p.name] = w

    output = W.Output()
    status = W.HTML()

    def read(param, widget):
        return from_text(param, widget.value) if isinstance(widget, W.Text) else widget.value

    def redraw(_change=None):
        try:
            dist.set_params(**{p.name: read(p, controls[p.name]) for p in dist._spec.params})
        except (TypeError, ValueError) as error:
            status.value = f"<span style='color:#d62728'>⚠ {html.escape(str(error))}</span>"
            return
        status.value = ""
        with output:
            output.clear_output(wait=True)
            try:
                dist._plot_static(type=type, show=True, **plot_kwargs)
            except (NotImplementedError, ValueError) as error:
                plt.close("all")
                status.value = f"<span style='color:#d62728'>⚠ {html.escape(str(error))}</span>"

    for w in controls.values():
        w.observe(redraw, names="value")
    display(W.VBox([*controls.values(), status, output]))
    redraw()
    return None


def _explore_matplotlib(dist, type, sliders, plot_kwargs):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, TextBox

    from ._plotting import style_context

    dark = plot_kwargs.get("dark", False)
    params = dist._spec.params
    width, height = plot_kwargs.pop("figsize", (8, 5) if not dist.is_multivariate else (6, 5))
    row = 0.45  # inches per control
    controls_height = row * len(params) + 0.35
    fig_height = height + controls_height

    with style_context(dark):
        fig = plt.figure(figsize=(width, fig_height))
        bottom = (controls_height + 0.55) / fig_height
        rect = [0.12, bottom, 0.8, 1 - bottom - 0.08]
        ax = fig.add_axes(rect, projection="3d") if dist.is_multivariate else fig.add_axes(rect)
        widgets = {}
        for i, p in enumerate(params):
            y = (controls_height - row * (i + 1)) / fig_height
            wax = fig.add_axes([0.25, y, 0.55, 0.6 * row / fig_height])
            value = dist._params[p.name]
            rng = slider_range(p, value, sliders.get(p.name))
            if rng is not None:
                lo, hi, step = rng
                widgets[p.name] = Slider(wax, p.symbol, lo, hi, valinit=value,
                                         valstep=1 if p.kind == "int" else None)
            else:
                widgets[p.name] = TextBox(wax, p.symbol + " ", initial=to_text(value))

    def read(param, widget):
        if isinstance(widget, Slider):
            return int(round(widget.val)) if param.kind == "int" else widget.val
        return from_text(param, widget.text)

    def update(_event=None):
        with style_context(dark):
            try:
                dist.set_params(**{p.name: read(p, widgets[p.name]) for p in params})
                ax.clear()
                dist._plot_static(type=type, ax=ax, show=False, **plot_kwargs)
            except (TypeError, ValueError, NotImplementedError) as error:
                ax.set_title(f"⚠ {error}", color="#d62728")
        fig.canvas.draw_idle()

    for w in widgets.values():
        if isinstance(w, Slider):
            w.on_changed(update)
        else:
            w.on_submit(update)
    fig._probability_distributions_widgets = widgets  # keep the widgets alive (and responsive)
    update()
    plt.show()
    return ax
