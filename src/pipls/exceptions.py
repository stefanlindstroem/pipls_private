"""Public warning categories for Pi-PLS."""


class StatisticalSupportWarning(UserWarning):
    """Warning for a configuration below the documented support margin.

    Direct fixed fits warn below three observations per retained predictor-rank
    direction. Path searches warn when their configured support rule permits fewer
    than five supplied observations per retained direction.
    """
