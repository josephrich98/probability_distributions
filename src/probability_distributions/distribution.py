"""The :class:`Distribution` class."""

from __future__ import annotations

import logging
import math
import os
import textwrap
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import catalog  # noqa: F401  (registers the distributions)
from ._spec import REGISTRY, DistSpec, _fmt, lookup

logger = logging.getLogger("probability_distributions")

_INFO_WIDTH = 100
_INFO_LABEL = 18


class Distribution:
    """A probability distribution with fixed parameters.

    Parameters
    ----------
    name
        Distribution name, e.g. ``"binomial"``, ``"normal"``, ``"chi2"``. Case and ``-``/``_``/spaces are
        ignored, and common aliases are accepted (see :func:`available`).
    *args, **kwargs
        Parameter values, positionally in the order listed by ``Distribution(name).param_names`` or by name.
        Any parameter not given takes its default value. Common alternative parameterizations are accepted
        as keywords and converted, e.g. ``Distribution("gamma", k=2, theta=3)`` is shape α = 2 and rate
        β = 1/3, and ``Distribution("exponential", scale=2)`` is rate λ = 0.5 (see :meth:`info`).

    Examples
    --------
    >>> d = Distribution("binomial", n=20, p=0.3)
    >>> d.pmf(6)  # doctest: +ELLIPSIS
    0.19...
    >>> d.expected_value()
    6.0
    """

    def __init__(self, name: str, *args: Any, **kwargs: Any):
        spec, preset = lookup(name)
        self._spec: DistSpec = spec
        names = [p.name for p in spec.params]
        if len(args) > len(names):
            raise TypeError(f"{spec.display} takes at most {len(names)} parameters ({', '.join(names)}), "
                            f"got {len(args)}")
        kwargs = self._translate(kwargs)
        values: Dict[str, Any] = {p.name: p.default for p in spec.params}
        values.update(preset)
        for param_name, arg in zip(names, args):
            if param_name in kwargs:
                raise TypeError(f"parameter {param_name!r} given both positionally and by name")
            values[param_name] = arg
        values.update(kwargs)
        self._params: Dict[str, Any] = {}
        self._frozen: Any = None
        self._assign(values)

    # ------------------------------------------------------------------ parameters
    def _translate(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert alternative parameter keywords (e.g. ``theta``) to the canonical ones (e.g. ``beta``)."""
        translated: Dict[str, Any] = {}
        sources: Dict[str, str] = {}
        for key, value in kwargs.items():
            target = key
            if key in self._spec.alt_params and key not in self.param_names:
                target, convert, _ = self._spec.alt_params[key]
                try:
                    value = convert(value)
                except (ZeroDivisionError, TypeError):
                    raise ValueError(f"invalid value {key}={value!r}") from None
            if target in translated:
                raise TypeError(f"{sources[target]!r} and {key!r} both set {target!r}; give only one")
            translated[target] = value
            sources[target] = key
        unknown = set(translated) - set(self.param_names)
        if unknown:
            alternatives = f" (or {', '.join(self._spec.alt_params)})" if self._spec.alt_params else ""
            raise TypeError(f"unknown parameter(s) {sorted(unknown)} for {self._spec.display}; "
                            f"parameters are {self.param_names}{alternatives}")
        return translated

    def _assign(self, values: Dict[str, Any]) -> None:
        coerced = {p.name: p.coerce(values[p.name]) for p in self._spec.params}
        if self._spec.normalize is not None:
            coerced = self._spec.normalize(coerced)
        if self._spec.check is not None:
            self._spec.check(coerced)
        self._frozen = self._spec.scipy(**coerced)
        self._params = coerced

    def set_params(self, **kwargs: Any) -> "Distribution":
        """Change some parameters in place (validated atomically). Returns ``self``."""
        self._assign({**self._params, **self._translate(kwargs)})
        return self

    def copy(self, **kwargs: Any) -> "Distribution":
        """Return a new distribution of the same family, optionally with some parameters changed."""
        return Distribution(self._spec.name, **self._params).set_params(**kwargs)

    @property
    def params(self) -> Dict[str, Any]:
        """The current parameter values (a copy)."""
        return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self._params.items()}

    @property
    def param_names(self) -> List[str]:
        return [p.name for p in self._spec.params]

    def __getattr__(self, item: str) -> Any:
        # Parameters are readable as attributes: Distribution("binomial", n=5).n == 5
        params = self.__dict__.get("_params")
        if params is not None and item in params:
            return params[item]
        raise AttributeError(f"{type(self).__name__!s} object has no attribute {item!r}")

    # ------------------------------------------------------------------ descriptive properties
    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def display_name(self) -> str:
        return self._spec.display

    @property
    def kind(self) -> str:
        """``"discrete"`` or ``"continuous"``."""
        return self._spec.kind

    @property
    def is_discrete(self) -> bool:
        return self._spec.kind == "discrete"

    @property
    def is_continuous(self) -> bool:
        return self._spec.kind == "continuous"

    @property
    def is_multivariate(self) -> bool:
        return self._spec.multivariate

    @property
    def scipy(self) -> Any:
        """The underlying frozen :mod:`scipy.stats` distribution."""
        return self._frozen

    def param_string(self, symbols: bool = True) -> str:
        """``"n=10, p=0.3"`` style summary of the parameters."""
        return ", ".join(f"{p.symbol if symbols else p.name}={_fmt(self._params[p.name])}"
                         for p in self._spec.params)

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={_fmt(v) if isinstance(v, np.ndarray) else repr(v)}" for k, v in self._params.items())
        return f"Distribution({self._spec.name!r}, {args})"

    def __str__(self) -> str:
        return f"{self._spec.display}({self.param_string()})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Distribution) or other._spec is not self._spec:
            return NotImplemented
        return all(np.array_equal(self._params[k], other._params[k]) for k in self._params)

    __hash__ = None  # mutable (set_params)

    # ------------------------------------------------------------------ PMF / PDF / CDF / MGF
    def pmf(self, x: Any = None) -> Any:
        """Probability mass function P(X = x).

        With no argument, returns the formula as a string. For a continuous distribution this logs a
        warning and evaluates the PDF instead (``pmf`` and ``pdf`` are aliases).
        """
        if self.is_continuous:
            logger.warning("%s is continuous, so it has a PDF rather than a PMF; using the PDF.", self)
        return self._density(x)

    def pdf(self, x: Any = None) -> Any:
        """Probability density function f(x).

        With no argument, returns the formula as a string. For a discrete distribution this logs a
        warning and evaluates the PMF instead (``pmf`` and ``pdf`` are aliases).
        """
        if self.is_discrete:
            logger.warning("%s is discrete, so it has a PMF rather than a PDF; using the PMF.", self)
        return self._density(x)

    def _density(self, x: Any) -> Any:
        if x is None:
            return self._spec.density
        if self.is_multivariate:
            x = np.asarray(x, dtype=float)
            if self._spec.name == "dirichlet":
                x = np.moveaxis(x, -1, 0)  # scipy wants the components first
            return self._frozen.pmf(x) if self.is_discrete else self._frozen.pdf(x)
        return self._frozen.pmf(x) if self.is_discrete else self._frozen.pdf(x)

    def cdf(self, x: Any = None) -> Any:
        """Cumulative distribution function F(x) = P(X ≤ x). With no argument, returns the formula."""
        if x is None:
            return self._spec.cdf
        if self.is_multivariate:
            if self._spec.name == "multivariate_normal":
                return self._frozen.cdf(np.asarray(x, dtype=float))
            raise NotImplementedError(f"CDF is not implemented for the {self._spec.display} distribution")
        if self.is_discrete:
            x = np.floor(x)  # F(x) = F(⌊x⌋); some scipy distributions return nan between integers
        return self._frozen.cdf(x)

    def mgf(self, t: Any = None) -> Any:
        """Moment generating function M(t) = E[e^(tX)]. With no argument, returns the formula.

        Uses the closed form where one exists and numerical integration/summation otherwise. Returns ``nan``
        (with a warning) where the MGF does not exist. For multivariate distributions ``t`` is a vector.
        """
        if t is None:
            return self._spec.mgf
        P = self._params
        if self.is_multivariate:
            if self._spec.mgf_fn is None:
                raise NotImplementedError(f"No MGF implementation for the {self._spec.display} distribution")
            return self._spec.mgf_fn(P, t)

        t_arr = np.asarray(t, dtype=float)
        flat = t_arr.ravel()
        lo, hi = self._spec.mgf_domain(P)
        exists = (flat > lo) & (flat < hi)
        result = np.full(flat.shape, np.nan)
        result[flat == 0] = 1.0
        todo = exists & (flat != 0)
        if np.any(todo):
            if self._spec.mgf_fn is not None:
                with np.errstate(over="ignore"):
                    result[todo] = self._spec.mgf_fn(P, flat[todo])
            else:
                result[todo] = [self._frozen.expect(lambda x, s=s: np.exp(s * x)) for s in flat[todo]]
        missing = ~exists & (flat != 0)
        if np.any(missing):
            logger.warning("The MGF of %s does not exist at t=%s (it exists only for %s < t < %s); returning nan.",
                           self, _fmt(flat[missing][0]), _fmt(lo), _fmt(hi))
        result = result.reshape(t_arr.shape)
        return float(result) if result.ndim == 0 else result

    # ------------------------------------------------------------------ summaries
    def parameter_space(self) -> str:
        """The set of valid parameter values."""
        return self._spec.parameter_space

    def support(self) -> str:
        """The support (event space) as a formula, plus its value for the current parameters."""
        general = self._spec.support
        if self.is_multivariate:
            return general
        return f"{general}   (here: {self._support_string()})"

    def support_bounds(self) -> Tuple[float, float]:
        """Numeric (lower, upper) bounds of the support (univariate distributions only)."""
        if self.is_multivariate:
            raise NotImplementedError("support_bounds is only defined for univariate distributions")
        lo, hi = self._frozen.support()
        return float(lo), float(hi)

    def _support_string(self) -> str:
        lo, hi = self.support_bounds()
        if self.is_discrete:
            if math.isfinite(hi) and hi - lo <= 3:
                return "{" + ", ".join(str(int(v)) for v in np.arange(lo, hi + 1)) + "}"
            tail = f"{int(hi)}" if math.isfinite(hi) else ""
            return "{" + f"{int(lo)}, {int(lo) + 1}, …" + (f", {tail}" if tail else "") + "}"
        return f"{_fmt(lo)} to {_fmt(hi)}"

    def expected_value(self, formula: bool = False) -> Any:
        """E[X] for the current parameters (``formula=True`` returns the general formula instead).

        Returns ``inf`` or ``nan`` when the mean is infinite or undefined (e.g. Cauchy).
        """
        if formula:
            return self._spec.mean
        if self._spec.mean_fn is not None:
            return self._spec.mean_fn(self._params)
        return float(self._frozen.mean())

    mean = expected_value

    def variance(self, formula: bool = False) -> Any:
        """Var(X) for the current parameters (``formula=True`` returns the general formula instead).

        Multivariate distributions return the covariance matrix.
        """
        if formula:
            return self._spec.variance
        if self._spec.var_fn is not None:
            return self._spec.var_fn(self._params)
        return float(self._frozen.var())

    def std(self) -> Any:
        """Standard deviation sqrt(Var(X)) (per component for multivariate distributions)."""
        var = self.variance()
        return np.sqrt(np.diag(var)) if np.ndim(var) == 2 else math.sqrt(var)

    def story(self) -> str:
        """The generative 'story' behind the distribution and what each parameter means."""
        return self._spec.story

    def use_cases(self) -> List[str]:
        """Where the distribution is used outside of its story."""
        return list(self._spec.use_cases)

    def special_facts(self) -> List[str]:
        """Relationships to other distributions and notable properties."""
        return list(self._spec.special_facts)

    def describe(self) -> str:
        """Everything :meth:`info` prints, as a string."""
        density = self._spec.density_name
        lines = [f"{self}  —  {self.kind}{', multivariate' if self.is_multivariate else ''}",
                 "─" * min(_INFO_WIDTH, 60)]

        def add(label: str, text: str) -> None:
            lines.append(textwrap.fill(text, width=_INFO_WIDTH, initial_indent=f"{label + ':':<{_INFO_LABEL}}",
                                       subsequent_indent=" " * _INFO_LABEL))

        def add_list(label: str, items: Sequence[str]) -> None:
            for i, item in enumerate(items):
                prefix = f"{label + ':':<{_INFO_LABEL}}" if i == 0 else " " * _INFO_LABEL
                lines.append(textwrap.fill(f"• {item}", width=_INFO_WIDTH, initial_indent=prefix,
                                           subsequent_indent=" " * (_INFO_LABEL + 2)))

        add("Story", self.story())
        add_list("Parameters", [f"{p.symbol} = {p.description}  [here: {_fmt(self._params[p.name])}]"
                                for p in self._spec.params])
        if self._spec.alt_params:
            groups: Dict[str, List[str]] = {}
            for key, (_, _, note) in self._spec.alt_params.items():
                groups.setdefault(note, []).append(key)
            add("Also accepts", ";  ".join(f"{', '.join(keys)} ({note})" for note, keys in groups.items()))
        add("Parameter space", self.parameter_space())
        add("Support", self.support())
        add(density, self._spec.density)
        add("CDF", self._spec.cdf)
        add("Mean", f"{self._spec.mean}  =  {_value_string(self.expected_value())}")
        add("Variance", f"{self._spec.variance}  =  {_value_string(self.variance())}")
        add("MGF", self._spec.mgf)
        add_list("Use cases", self.use_cases())
        add_list("Special facts", self.special_facts())
        return "\n".join(lines)

    def info(self) -> None:
        """Print the story, parameters, parameter space, support, PMF/PDF, CDF, mean, variance, MGF,
        use cases and special facts."""
        print(self.describe())

    # ------------------------------------------------------------------ sampling
    def sample(self, n: int = 1, out: Optional[str] = None, seed: Any = None) -> Any:
        """Draw ``n`` random samples.

        Returns a single value when ``n == 1`` (a vector for multivariate distributions) and an array of
        shape ``(n,)`` (``(n, k)`` for multivariate) otherwise. If ``out`` is a path, the samples are also
        written to that text file, one sample per line (tab-separated components for multivariate).
        """
        if int(n) != n or n < 1:
            raise ValueError(f"n must be a positive integer, got {n!r}")
        n = int(n)
        rng = np.random.default_rng(seed)
        draws = np.asarray(self._frozen.rvs(size=n, random_state=rng))
        draws = draws.reshape(n, -1) if self.is_multivariate else draws.reshape(n)
        if out is not None:
            _makedirs_for(out)
            fmt = "%d" if np.issubdtype(draws.dtype, np.integer) else "%.10g"
            np.savetxt(out, draws, fmt=fmt, delimiter="\t")
        return draws[0] if n == 1 else draws

    # ------------------------------------------------------------------ plotting
    def plot(self, type: Optional[str] = None, out: Optional[str] = None, *, interactive: Optional[bool] = None,
             sliders: Optional[Dict[str, Tuple[float, float]]] = None, **kwargs: Any) -> Any:
        """Plot the PMF/PDF or CDF.

        Parameters
        ----------
        type
            ``"pmf"``, ``"pdf"`` (aliases; a warning is logged if the name doesn't match the distribution) or
            ``"cdf"``. Defaults to the PMF for discrete and the PDF for continuous distributions.
        out
            Path to save the figure to (format from the extension, e.g. ``.png``, ``.pdf``, ``.svg``).
        interactive
            Show a control per parameter to explore how the plot changes: a slider for parameters in a
            bounded range (e.g. p ∈ [0, 1]) and a text box otherwise. Uses ipywidgets in Jupyter and
            matplotlib widgets in a GUI window. Default: on when displaying (``out`` is None, ``show`` is not
            False, no ``ax``) in an environment that supports it. Moving a control updates this object's
            parameters in place.
        sliders
            Force sliders with the given ``(min, max)`` range for any parameter, e.g. ``{"mu": (-5, 5)}``.
        **kwargs
            Plot options shared with :func:`probability_distributions.utils.plot`: ``xmin``, ``xmax``,
            ``ymin``, ``ymax``, ``xlog``, ``ylog``, ``samples`` (int or data to overlay as a histogram),
            ``y`` (``"fraction"`` or ``"count"``), ``bins``, ``style`` (``"line"``, ``"bar"`` or ``"stem"``
            for discrete), ``plot_normal``, ``color``, ``label``, ``title``, ``xlabel``, ``ylabel``,
            ``legend_loc``, ``figsize``, ``dark``, ``ax``, ``show``, ``n_points``, ``seed``, ``dpi``.

        Returns
        -------
        The matplotlib Axes (``None`` for an ipywidgets display).
        """
        from . import _interactive

        if interactive is None:
            interactive = (out is None and kwargs.get("show") is not False and kwargs.get("ax") is None
                           and _interactive.mode() is not None)
        if interactive:
            return _interactive.explore(self, type=type, sliders=sliders or {}, **kwargs)
        return self._plot_static(type=type, out=out, **kwargs)

    def _plot_static(self, type: Optional[str] = None, out: Optional[str] = None, **kwargs: Any) -> Any:
        from . import _plotting

        if self.is_multivariate:
            return _plotting.plot_multivariate(self, type=type, out=out, **kwargs)
        color = kwargs.pop("color", None)
        label = kwargs.pop("label", None)
        samples = kwargs.pop("samples", None)
        return _plotting.plot_distributions(
            [self], type=type, out=out,
            colors=None if color is None else [color],
            labels=None if label is None else [label],
            samples=None if samples is None else ([samples] if not np.isscalar(samples) else samples),
            **kwargs,
        )


def _value_string(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return np.array2string(value, precision=4, separator=", ", suppress_small=True).replace("\n", "")
    if isinstance(value, float) and math.isnan(value):
        return "undefined"
    return _fmt(value)


def _makedirs_for(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)


def available(kind: Optional[str] = None) -> List[str]:
    """Names of all available distributions, optionally only ``"discrete"``, ``"continuous"`` or
    ``"multivariate"`` ones."""
    specs = REGISTRY.values()
    if kind == "multivariate":
        specs = [s for s in specs if s.multivariate]
    elif kind is not None:
        specs = [s for s in specs if s.kind == kind]
    return [s.name for s in specs]
