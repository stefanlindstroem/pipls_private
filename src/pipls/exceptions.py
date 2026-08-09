"""Public warning categories for Π-PLS."""

__all__ = ["PredictorRankSupportWarning"]


class PredictorRankSupportWarning(UserWarning):
    """Warning for low sample support relative to retained predictor rank.

    Direct fixed fits warn below three observations per retained predictor-rank
    direction. Path searches warn when their configured support rule permits fewer
    than five supplied observations per retained direction.
    """
