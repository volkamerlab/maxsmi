"""
Unit and regression test for the maxsmi package.
"""

# Import package, test suite, and other packages as needed
# import maxsmi
import pytest
import sys

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
    "smiles, solution",
    [
        ("C", ["C"]),
    ],
)
def test_no_augmentation(smiles, solution):
    smiles = no_augmentation(smiles)
    assert solution == smiles


@pytest.mark.parametrize(
    "smiles, aug_nb, solution",
    [
        ("CCC", 1, ["CCC"]),
    ],
)
def test_augmentation_with_duplication(smiles, aug_nb, solution):
    smiles = augmentation_with_duplication(smiles, aug_nb)
    assert solution == smiles


@pytest.mark.parametrize(
    "smiles, aug_nb, solution",
    [
        ("CCC", 10, ["CCC", "C(C)C"]),
    ],
)
def test_augmentation_without_duplication(smiles, aug_nb, solution):
    smiles = augmentation_without_duplication(smiles, aug_nb)
    assert solution == smiles


@pytest.mark.parametrize(
    "smiles, aug_nb, solution",
    [
        ("CCC", 1, ["CCC"]),
    ],
)
def test_augmentation_with_reduced_duplication(smiles, aug_nb, solution):
    smiles = augmentation_with_reduced_duplication(smiles, aug_nb)
    assert solution == smiles


@pytest.mark.parametrize(
    "smiles, aug_nb, solution",
    [
        ("CCC", 10, ["CCC", "C(C)C"]),
    ],
)
def test_augmentation_maximum_estimation(smiles, aug_nb, solution):
    smiles = augmentation_maximum_estimation(smiles, aug_nb)
    assert solution == smiles
