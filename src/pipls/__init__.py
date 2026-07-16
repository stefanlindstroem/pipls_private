"""Pi-PLS multivariate regression."""

from .decomposition import PiPLSDecomposition
from .exceptions import StatisticalSupportWarning
from .path import PiPLSPathCV
from .regression import PiPLSRegression
from .validation import PiPLSValidationReport

__all__ = [
    "PiPLSDecomposition",
    "PiPLSPathCV",
    "PiPLSRegression",
    "PiPLSValidationReport",
    "StatisticalSupportWarning",
    "__version__",
]

__version__ = "0.0.0"
