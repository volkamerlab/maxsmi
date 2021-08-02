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
)


def test_maxsmi_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "maxsmi" in sys.modules


####################
@pytest.mark.parametrize(
    "task, solution",
    [
        ("free_solv", 76),
        ("ESOL", 109),
        ("lipo", 268),
        ("lipophilicity", 268),
        ("chembl28", 246),
        ("affinity", 246),
        ("free solv", None),
    ],
)
def test_retrieve_longest_smiles_from_optimal_model(task, solution):
    length = retrieve_longest_smiles_from_optimal_model(task)
    assert solution == length


def test_unlabeled_smiles_max_length():
    assert unlabeled_smiles_max_length("CCC", 10) == None


def test_unlabeled_smiles_max_length_exception():
    with pytest.raises(Exception):
        assert unlabeled_smiles_max_length("CCC", 2)


def test_mixture_check():
    assert "CC" == mixture_check("CC")


def test_mixture_check_exception():
    with pytest.raises(Exception):
        assert mixture_check("C.C")
