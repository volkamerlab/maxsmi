"""
Unit and regression test for the maxsmi package.
"""

# Import package, test suite, and other packages as needed
# import maxsmi
import pytest
import sys
from maxsmi.utils import string_to_bool, augmentation_strategy
from maxsmi.augmentation_strategies import (
    no_augmentation,
    augmentation_with_duplication,
    augmentation_without_duplication,
    augmentation_with_reduced_duplication,
    augmentation_maximum_estimation,
)


def test_maxsmi_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "maxsmi" in sys.modules


####################
@pytest.mark.parametrize(
    "string, solution",
    [("yes", True), ("True", True), ("False", False), (True, True), (False, False)],
)
def test_string_to_bool(string, solution):
    result = string_to_bool(string)
    assert solution == result


def test_string_to_bool_exception():
    with pytest.raises(Exception):
        assert string_to_bool("Fasle")


@pytest.mark.parametrize(
    "string, solution",
    [
        ("no_augmentation", no_augmentation),
        ("augmentation_with_duplication", augmentation_with_duplication),
        ("augmentation_without_duplication", augmentation_without_duplication),
        (
            "augmentation_with_reduced_duplication",
            augmentation_with_reduced_duplication,
        ),
        ("augmentation_maximum_estimation", augmentation_maximum_estimation),
    ],
)
def test_augmentation_strategy(string, solution):
    result = augmentation_strategy(string)
    assert solution == result


def test_augmentation_strategy_exception():
    with pytest.raises(Exception):
        assert augmentation_strategy("augment")
