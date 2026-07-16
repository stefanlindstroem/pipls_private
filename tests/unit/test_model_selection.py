from __future__ import annotations

import numpy as np
import pytest

from pipls.model_selection import _max_predictor_rank


def test_max_predictor_rank_uses_ceiling_rule() -> None:
    assert (
        _max_predictor_rank(
            n_features=100,
            n_train_min=51,
            samples_per_predictor_rank=10,
        )
        == 6
    )


def test_max_predictor_rank_respects_feature_and_training_caps() -> None:
    assert (
        _max_predictor_rank(
            n_features=3,
            n_train_min=100,
            samples_per_predictor_rank=10,
        )
        == 3
    )
    assert (
        _max_predictor_rank(
            n_features=100,
            n_train_min=4,
            samples_per_predictor_rank=0.1,
        )
        == 4
    )


def test_smaller_training_fold_cannot_increase_rank_bound() -> None:
    complete_data_bound = _max_predictor_rank(
        n_features=30,
        n_train_min=50,
        samples_per_predictor_rank=10,
    )
    fold_bound = _max_predictor_rank(
        n_features=30,
        n_train_min=39,
        samples_per_predictor_rank=10,
    )

    assert complete_data_bound == 5
    assert fold_bound == 4


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("n_features", 0),
        ("n_train_min", 0),
        ("samples_per_predictor_rank", 0.0),
        ("samples_per_predictor_rank", np.inf),
        ("samples_per_predictor_rank", True),
    ],
)
def test_max_predictor_rank_rejects_invalid_inputs(argument: str, value: object) -> None:
    kwargs: dict[str, object] = {
        "n_features": 10,
        "n_train_min": 20,
        "samples_per_predictor_rank": 10,
    }
    kwargs[argument] = value

    with pytest.raises(ValueError, match=argument):
        _max_predictor_rank(**kwargs)  # type: ignore[arg-type]
