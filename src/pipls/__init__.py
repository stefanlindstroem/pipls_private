"""Pi-PLS multivariate regression."""

from .component_path import (
    PiPLSComponentPath,
    PiPLSComponentResult,
    PiPLSPredictorRankProfile,
)
from .decomposition import PiPLSDecomposition
from .exceptions import StatisticalSupportWarning
from .path import PiPLSPathCV
from .regression import PiPLSRegression
from .validation import PiPLSValidationReport

__all__ = [
    "PiPLSComponentPath",
    "PiPLSComponentResult",
    "PiPLSPredictorRankProfile",
    "PiPLSDecomposition",
    "PiPLSPathCV",
    "PiPLSRegression",
    "PiPLSValidationReport",
    "StatisticalSupportWarning",
    "__version__",
]

__version__ = "0.0.0"
