"""Specification objects describing each distribution in the catalog."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

INF = math.inf

Params = Dict[str, Any]


@dataclass(frozen=True)
class Param:
    """A single parameter of a distribution.

    ``kind`` is one of ``"int"``, ``"float"``, ``"vector"`` or ``"matrix"``. Bounds apply
    elementwise to vectors; matrices are not bounds-checked (their distribution's ``check`` does it).
    """

    name: str
    symbol: str
    description: str
    default: Any
    kind: str = "float"
    lower: float = -INF
    upper: float = INF
    lower_inclusive: bool = False
    upper_inclusive: bool = False

    @property
    def bounded(self) -> bool:
        """True if the parameter lives in a finite interval (so it can be a slider)."""
        return self.kind in ("int", "float") and math.isfinite(self.lower) and math.isfinite(self.upper)

    def coerce(self, value: Any) -> Any:
        """Convert ``value`` to this parameter's type and check its bounds."""
        if self.kind == "int":
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
                raise ValueError(f"{self.name} must be an integer, got {value!r}")
            if float(value) != int(value):
                raise ValueError(f"{self.name} must be an integer, got {value!r}")
            value = int(value)
        elif self.kind == "float":
            if not np.isscalar(value) or isinstance(value, str):
                raise ValueError(f"{self.name} must be a number, got {value!r}")
            value = float(value)
        elif self.kind == "vector":
            value = np.asarray(value, dtype=float)
            if value.ndim != 1 or value.size == 0:
                raise ValueError(f"{self.name} must be a non-empty 1-D sequence of numbers")
        elif self.kind == "matrix":
            value = np.asarray(value, dtype=float)
            if value.ndim != 2 or value.shape[0] != value.shape[1]:
                raise ValueError(f"{self.name} must be a square 2-D matrix")
        else:  # pragma: no cover - programming error
            raise AssertionError(self.kind)

        if self.kind in ("int", "float", "vector"):
            arr = np.atleast_1d(value)
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{self.name} must be finite")
            low_ok = arr >= self.lower if self.lower_inclusive else arr > self.lower
            high_ok = arr <= self.upper if self.upper_inclusive else arr < self.upper
            if not (np.all(low_ok) and np.all(high_ok)):
                raise ValueError(f"{self.name}={_fmt(value)} is outside its allowed range {self.interval()}")
        return value

    def interval(self) -> str:
        left = "[" if self.lower_inclusive else "("
        right = "]" if self.upper_inclusive else ")"
        lo = "−∞" if self.lower == -INF else _fmt(self.lower)
        hi = "∞" if self.upper == INF else _fmt(self.upper)
        return f"{left}{lo}, {hi}{right}"


@dataclass
class DistSpec:
    """Everything the package knows about one distribution family."""

    name: str
    display: str
    kind: str  # "discrete" or "continuous"
    params: Tuple[Param, ...]
    scipy: Callable[..., Any]  # (**params) -> frozen scipy distribution
    parameter_space: str
    support: str
    density: str  # PMF or PDF formula
    cdf: str
    mean: str
    variance: str
    mgf: str
    story: str
    use_cases: Sequence[str] = ()
    special_facts: Sequence[str] = ()
    aliases: Sequence[str] = ()
    check: Optional[Callable[[Params], None]] = None  # cross-parameter validation
    normalize: Optional[Callable[[Params], Params]] = None  # e.g. append the remaining probability
    mgf_fn: Optional[Callable[[Params, np.ndarray], np.ndarray]] = None  # closed form
    mgf_domain: Callable[[Params], Tuple[float, float]] = lambda P: (-INF, INF)
    xrange: Optional[Callable[[Params, Any], Tuple[float, float]]] = None  # (params, frozen) -> (lo, hi)
    style: str = "line"  # default style for discrete plots: "line" or "bar"
    multivariate: bool = False
    mean_fn: Optional[Callable[[Params], Any]] = None
    var_fn: Optional[Callable[[Params], Any]] = None
    support_bounds: Optional[Callable[[Params], Tuple[float, float]]] = None
    presets: Params = field(default_factory=dict)  # fixed parameter values (used by aliases like standard_normal)
    # Alternative parameterizations: keyword -> (canonical parameter, converter, description)
    alt_params: Dict[str, Tuple[str, Callable[[Any], Any], str]] = field(default_factory=dict)

    @property
    def density_name(self) -> str:
        return "PMF" if self.kind == "discrete" else "PDF"


REGISTRY: Dict[str, DistSpec] = {}
ALIASES: Dict[str, Tuple[str, Params]] = {}


def _key(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def register(spec: DistSpec) -> DistSpec:
    REGISTRY[spec.name] = spec
    for alias in spec.aliases:
        ALIASES[_key(alias)] = (spec.name, {})
    return spec


def register_alias(alias: str, name: str, **preset: Any) -> None:
    """Register ``alias`` as ``name`` with some parameters fixed by default."""
    ALIASES[_key(alias)] = (name, preset)


def lookup(name: str) -> Tuple[DistSpec, Params]:
    key = _key(name)
    if key in REGISTRY:
        return REGISTRY[key], {}
    if key in ALIASES:
        target, preset = ALIASES[key]
        return REGISTRY[target], dict(preset)
    raise ValueError(f"Unknown distribution {name!r}. Available: {', '.join(sorted(REGISTRY))}")


def same(name: str) -> Tuple[str, Callable[[Any], Any], str]:
    """An alternative keyword that is just another name for ``name``."""
    return name, lambda v: v, f"= {name}"


def inverse(name: str, meaning: str) -> Tuple[str, Callable[[Any], Any], str]:
    """An alternative keyword whose reciprocal is ``name`` (e.g. scale = 1/rate)."""
    return name, lambda v: 1 / v, f"{meaning}; {name} = 1/{meaning.split()[-1]}"


def _fmt(value: Any) -> str:
    """Compact number formatting for labels and messages."""
    if isinstance(value, np.ndarray):
        if value.ndim == 2:
            return "[" + "; ".join(", ".join(f"{v:g}" for v in row) for row in value) + "]"
        return "[" + ", ".join(f"{v:g}" for v in value) + "]"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if math.isinf(value):
            return "∞" if value > 0 else "−∞"
        return f"{value:g}"
    return str(value)
