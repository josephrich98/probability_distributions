"""Probability distributions: stories, formulas, facts, sampling and (interactive) plots.

>>> from probability_distributions import Distribution
>>> d = Distribution("poisson", lam=4)
>>> d.info()                      # story, support, PMF, CDF, mean, variance, MGF, facts...
>>> d.plot()                      # PMF (with a control for λ when displayed interactively)
>>> d.plot("cdf", out="poisson_cdf.png")
>>> d.sample(5, seed=0)
"""

from . import utils
from .distribution import Distribution, available

__all__ = ["Distribution", "available", "utils"]
__version__ = "0.1.0"
