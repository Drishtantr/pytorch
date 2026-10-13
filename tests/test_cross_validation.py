"""Tests for ml_utils/cross_validation.py."""

import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml_utils"))

from cross_validation import cross_validate, k_folds  # noqa: E402


@pytest.mark.parametrize("n_samples, k", [(10, 2), (10, 5), (10, 10), (11, 3), (23, 4)])
def test_k_folds_yields_exactly_k_folds(n_samples, k):
    assert len(list(k_folds(n_samples, k, seed=0))) == k


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("n_samples, k", [(10, 5), (11, 3), (23, 4), (7, 7)])
def test_validation_folds_partition_the_samples(n_samples, k, shuffle):
    val_indices = [
        i for _, val in k_folds(n_samples, k, shuffle=shuffle, seed=1) for i in val
    ]

    assert len(val_indices) == n_samples  # no index in two folds
    assert sorted(val_indices) == list(range(n_samples))  # every index covered


@pytest.mark.parametrize("n_samples, k", [(10, 5), (11, 3), (23, 4)])
def test_train_and_validation_are_disjoint_and_complete(n_samples, k):
    for train, val in k_folds(n_samples, k, seed=2):
        assert not set(train) & set(val)
        assert sorted(train + val) == list(range(n_samples))


@pytest.mark.parametrize(
    "n_samples, k, expected_sizes",
    [
        (11, 3, [4, 4, 3]),
        (23, 4, [6, 6, 6, 5]),
        (13, 5, [3, 3, 3, 2, 2]),
        (10, 5, [2] * 5),
    ],
)
def test_remainder_goes_to_earliest_folds(n_samples, k, expected_sizes):
    sizes = [len(val) for _, val in k_folds(n_samples, k, seed=3)]

    assert sizes == expected_sizes
    assert max(sizes) - min(sizes) <= 1


def test_unshuffled_folds_are_contiguous():
    assert [val for _, val in k_folds(5, 2, shuffle=False)] == [[0, 1, 2], [3, 4]]


def test_same_seed_same_folds_different_seed_different_folds():
    first = list(k_folds(20, 4, seed=42))

    assert list(k_folds(20, 4, seed=42)) == first
    assert list(k_folds(20, 4, seed=43)) != first


@pytest.mark.parametrize("k", [-1, 0, 1, 11])
def test_invalid_k_raises_at_call_time(k):
    with pytest.raises(ValueError):
        k_folds(10, k)


def test_cross_validate_returns_scores_mean_and_std():
    features = [[float(x)] for x in range(12)]
    targets = [2.0 * x for x in range(12)]
    seen_val_rows = []

    def fit_predict(train_x, train_y, val_x, val_y):
        assert len(train_x) == len(train_y)
        assert len(val_x) == len(val_y)
        assert not {row[0] for row in train_x} & {row[0] for row in val_x}
        seen_val_rows.extend(row[0] for row in val_x)
        return sum(val_y) / len(val_y)

    result = cross_validate(features, targets, fit_predict, k=4, seed=0)

    assert len(result.scores) == 4
    assert result.mean == pytest.approx(statistics.fmean(result.scores))
    assert result.std == pytest.approx(statistics.pstdev(result.scores))
    assert sorted(seen_val_rows) == [float(x) for x in range(12)]

    scores, mean, std = result
    assert (scores, mean, std) == (result.scores, result.mean, result.std)


def test_cross_validate_is_reproducible_with_seed():
    features = list(range(15))
    targets = [x * x for x in range(15)]

    def fit_predict(train_x, train_y, val_x, val_y):
        return float(sum(val_y))

    assert cross_validate(
        features, targets, fit_predict, k=3, seed=7
    ) == cross_validate(features, targets, fit_predict, k=3, seed=7)


def test_cross_validate_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        cross_validate([1, 2, 3], [1, 2], lambda *_: 0.0, k=2)


def test_cross_validate_rejects_invalid_k():
    with pytest.raises(ValueError):
        cross_validate([1, 2, 3], [1, 2, 3], lambda *_: 0.0, k=4)
