"""Tests for :mod:`ml_utils.data`.

Each test pins one property the helpers promise, so a regression in the
batching or scaling math shows up as a named failure rather than a model that
quietly trains on the wrong rows.
"""

import math
import random

import pytest

from ml_utils import data


def linear_dataset(n):
    """``n`` rows where every target is derivable from its own feature."""
    features = list(range(n))
    targets = [v * 10 for v in features]
    return features, targets


class TestTrainValSplit:
    def test_rows_keep_their_own_targets(self):
        features, targets = linear_dataset(50)
        train_x, train_y, val_x, val_y = data.train_val_split(features, targets)

        for x, y in zip(train_x + val_x, train_y + val_y, strict=True):
            assert y == x * 10

    def test_split_is_a_partition(self):
        features, targets = linear_dataset(50)
        train_x, _, val_x, _ = data.train_val_split(features, targets)

        assert len(train_x) + len(val_x) == 50
        assert not set(train_x) & set(val_x), "a row landed in both splits"
        assert set(train_x) | set(val_x) == set(features)

    def test_val_fraction_sizes_the_split(self):
        features, targets = linear_dataset(100)
        _, _, val_x, _ = data.train_val_split(features, targets, val_fraction=0.3)

        assert len(val_x) == 30

    def test_inputs_are_not_modified(self):
        features, targets = linear_dataset(20)
        data.train_val_split(features, targets)

        assert features == list(range(20))
        assert targets == [v * 10 for v in range(20)]

    def test_seeding_is_reproducible(self):
        features, targets = linear_dataset(40)

        data.set_seed(7)
        first = data.train_val_split(features, targets)
        data.set_seed(7)
        second = data.train_val_split(features, targets)

        assert first == second

    def test_does_not_disturb_global_random_state(self):
        random.seed(123)
        expected = [random.random() for _ in range(3)]

        random.seed(123)
        data.set_seed(999)
        features, targets = linear_dataset(20)
        data.train_val_split(features, targets)

        assert [random.random() for _ in range(3)] == expected

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            data.train_val_split([1, 2, 3], [1, 2])

    @pytest.mark.parametrize("fraction", [-0.1, 1.0, 1.5])
    def test_out_of_range_fraction_raises(self, fraction):
        features, targets = linear_dataset(10)
        with pytest.raises(ValueError, match="val_fraction"):
            data.train_val_split(features, targets, val_fraction=fraction)


class TestIterBatches:
    @pytest.mark.parametrize(
        "n,batch_size", [(10, 3), (12, 3), (32, 32), (64, 32), (100, 32), (1, 8)]
    )
    def test_every_row_is_visited_exactly_once(self, n, batch_size):
        features, targets = linear_dataset(n)
        seen = [
            row
            for batch_x, _ in data.iter_batches(features, targets, batch_size)
            for row in batch_x
        ]

        assert sorted(seen) == features

    @pytest.mark.parametrize(
        "n,batch_size", [(10, 3), (12, 3), (32, 32), (64, 32), (100, 32)]
    )
    def test_num_batches_matches_the_loop(self, n, batch_size):
        features, targets = linear_dataset(n)

        for drop_last in (False, True):
            actual = len(
                list(
                    data.iter_batches(
                        features, targets, batch_size, drop_last=drop_last
                    )
                )
            )
            assert actual == data.num_batches(features, batch_size, drop_last)

    def test_batches_keep_rows_paired_with_targets(self):
        features, targets = linear_dataset(37)

        for batch_x, batch_y in data.iter_batches(features, targets, 8):
            assert [x * 10 for x in batch_x] == batch_y

    def test_exact_multiple_yields_full_batches(self):
        features, targets = linear_dataset(32)
        batches = list(data.iter_batches(features, targets, 32))

        assert len(batches) == 1
        assert len(batches[0][0]) == 32

    def test_drop_last_discards_only_the_short_batch(self):
        features, targets = linear_dataset(10)
        batches = list(data.iter_batches(features, targets, 3, drop_last=True))

        assert len(batches) == 3
        assert all(len(batch_x) == 3 for batch_x, _ in batches)

    def test_shuffle_order_is_reproducible_under_set_seed(self):
        features, targets = linear_dataset(40)

        def order():
            data.set_seed(3)
            return [bx for bx, _ in data.iter_batches(features, targets, 8)]

        assert order() == order()

    def test_shuffle_actually_reorders(self):
        features, targets = linear_dataset(100)
        data.set_seed(0)
        seen = [
            row
            for bx, _ in data.iter_batches(features, targets, 10, shuffle=True)
            for row in bx
        ]

        assert seen != features
        assert sorted(seen) == features

    def test_unshuffled_preserves_order(self):
        features, targets = linear_dataset(10)
        seen = [
            row
            for bx, _ in data.iter_batches(features, targets, 3, shuffle=False)
            for row in bx
        ]

        assert seen == features

    def test_empty_dataset_yields_nothing(self):
        assert list(data.iter_batches([], [], 8)) == []

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            list(data.iter_batches([1, 2, 3], [1, 2], 2))

    @pytest.mark.parametrize("batch_size", [0, -1])
    def test_bad_batch_size_raises(self, batch_size):
        features, targets = linear_dataset(10)
        with pytest.raises(ValueError, match="batch_size"):
            list(data.iter_batches(features, targets, batch_size))
        with pytest.raises(ValueError, match="batch_size"):
            data.num_batches(features, batch_size)


class TestStandardize:
    def test_produces_zero_mean_unit_variance(self):
        scaled, _ = data.standardize([1.0, 2.0, 3.0, 4.0])

        assert math.isclose(sum(scaled) / len(scaled), 0.0, abs_tol=1e-12)
        variance = sum(v**2 for v in scaled) / len(scaled)
        assert math.isclose(variance, 1.0, rel_tol=1e-12)

    def test_input_is_not_modified(self):
        values = [1.0, 2.0, 3.0]
        scaled, _ = data.standardize(values)

        assert values == [1.0, 2.0, 3.0]
        assert scaled is not values

    def test_stats_replay_on_another_split(self):
        train = [1.0, 2.0, 3.0, 4.0]
        _, stats = data.standardize(train)
        mean, std = stats

        assert data.apply_standardizer([5.0], stats) == [(5.0 - mean) / std]

    def test_constant_feature_does_not_raise(self):
        scaled, (mean, std) = data.standardize([5.0, 5.0, 5.0])

        assert scaled == [0.0, 0.0, 0.0]
        assert (mean, std) == (5.0, 0.0)

    def test_no_state_leaks_between_calls(self):
        _, first = data.standardize([1.0, 2.0, 3.0])
        _, second = data.standardize([10.0, 20.0, 30.0])

        assert first != second
        assert first == data.fit_standardizer([1.0, 2.0, 3.0])

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty"):
            data.fit_standardizer([])


class TestAverageMeter:
    def test_weights_by_sample_count(self):
        meter = data.AverageMeter()
        meter.update(1.0, n=32)
        meter.update(4.0, n=8)

        assert math.isclose(meter.average, (1.0 * 32 + 4.0 * 8) / 40)

    def test_matches_the_mean_of_the_raw_values(self):
        values = [0.5, 1.5, 2.5, 3.5, 4.0]
        meter = data.AverageMeter()
        for value in values:
            meter.update(value)

        assert math.isclose(meter.average, sum(values) / len(values))

    def test_uneven_final_batch(self):
        """A short last batch must not count as much as a full one."""
        meter = data.AverageMeter()
        for _ in range(3):
            meter.update(2.0, n=32)
        meter.update(10.0, n=4)

        assert math.isclose(meter.average, (2.0 * 96 + 10.0 * 4) / 100)

    def test_empty_meter_is_zero(self):
        assert data.AverageMeter().average == 0.0

    def test_reset_clears_state(self):
        meter = data.AverageMeter()
        meter.update(5.0, n=10)
        meter.reset()

        assert meter.average == 0.0
        assert meter.count == 0

    def test_zero_count_update_is_ignored(self):
        meter = data.AverageMeter()
        meter.update(3.0, n=0)

        assert meter.average == 0.0

    def test_negative_count_raises(self):
        meter = data.AverageMeter()
        with pytest.raises(ValueError, match="non-negative"):
            meter.update(1.0, n=-1)


def test_end_to_end_epoch():
    """Split, batch and meter the whole dataset without losing or mixing rows."""
    data.set_seed(0)
    features, targets = linear_dataset(101)

    train_x, train_y, val_x, val_y = data.train_val_split(features, targets)
    assert len(train_x) + len(val_x) == 101

    stats = data.fit_standardizer(train_x)
    scaled_train = data.apply_standardizer(train_x, stats)
    scaled_val = data.apply_standardizer(val_x, stats)
    assert len(scaled_train) == len(train_x)
    assert len(scaled_val) == len(val_x)

    meter = data.AverageMeter()
    rows = 0
    for batch_x, batch_y in data.iter_batches(train_x, train_y, batch_size=16):
        assert [x * 10 for x in batch_x] == batch_y
        rows += len(batch_x)
        meter.update(1.0, n=len(batch_x))

    assert rows == len(train_x)
    assert math.isclose(meter.average, 1.0)
    assert meter.count == len(train_x)
