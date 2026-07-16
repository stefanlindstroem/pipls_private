"""Pi-PLS multivariate regression."""

from .decomposition import PiPLSDecomposition
from .exceptions import StatisticalSupportWarning
from .path import PiPLSPathCV
from .regression import PiPLSRegression

__all__ = [
    "PiPLSDecomposition",
    "PiPLSPathCV",
    "PiPLSRegression",
    "StatisticalSupportWarning",
    "__version__",
]

__version__ = "0.0.0"
