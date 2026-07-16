"""Pi-PLS multivariate regression."""

from .exceptions import StatisticalSupportWarning
from .regression import PiPLSRegression

__all__ = ["PiPLSRegression", "StatisticalSupportWarning", "__version__"]

__version__ = "0.0.0"
