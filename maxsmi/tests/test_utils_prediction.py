"""
Unit and regression test for the maxsmi package.
"""

# Import package, test suite, and other packages as needed
# import maxsmi
import pytest
import sys
from maxsmi.utils_prediction import (
    retrieve_longest_smiles_from_optimal_model,
    unlabeled_smiles_max_length,
    mixture_check,
    character_check,
)


def test_maxsmi_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "maxsmi" in sys.modules


####################
@pytest.mark.parametrize(
    "task, solution",
    [
        ("FreeSolv", 76),
        ("ESOL", 109),
        ("lipo", 268),
        ("lipophilicity", 268),
        ("chembl28", 246),
        ("affinity", 246),
        ("FreeSolv", None),
    ],
)
def test_retrieve_longest_smiles_from_optimal_model(task, solution):
    length = retrieve_longest_smiles_from_optimal_model(task)
    assert solution == length


@pytest.mark.parametrize(
    "smiles, length, solution",
    [
        ("CCC", 10, None),
    ],
)
def test_unlabeled_smiles_max_length(smiles, length, solution):
    result = unlabeled_smiles_max_length(smiles, length)
    assert solution == result


def test_unlabeled_smiles_max_length_exception():
    with pytest.raises(Exception):
        assert unlabeled_smiles_max_length("CCC", 2)


def test_mixture_check():
    assert "CC" == mixture_check("CC")


def test_mixture_check_exception():
    with pytest.raises(Exception):
        assert mixture_check("C.C")


@pytest.mark.parametrize(
    "smiles, solution",
    [
        ("CCC", "CCC"),
    ],
)
def test_character_check(smiles, solution):
    result = character_check(smiles)
    assert solution == result


def test_character_check_exception():
    with pytest.raises(Exception):
        assert character_check("CCçC")
