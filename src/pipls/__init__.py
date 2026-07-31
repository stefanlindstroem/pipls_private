"""Pi-PLS multivariate regression."""

from .component_path import (
    PiPLSComponentPath,
    PiPLSComponentResult,
    PiPLSPredictorRankProfile,
)
from .decomposition import PiPLSDecomposition
from .exceptions import PredictorRankSupportWarning
from .regression import PiPLSRegression
from .search import PiPLSSearchCV
from .validation import PiPLSValidationReport

__all__ = [
    "PiPLSComponentPath",
    "PiPLSComponentResult",
    "PiPLSPredictorRankProfile",
    "PiPLSDecomposition",
    "PiPLSRegression",
    "PiPLSSearchCV",
    "PiPLSValidationReport",
    "PredictorRankSupportWarning",
    "__version__",
]

__version__ = "0.0.0"
