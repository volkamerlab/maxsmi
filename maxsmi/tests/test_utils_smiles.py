"""
Unit and regression test for the maxsmi package.
"""

# Import package, test suite, and other packages as needed
# import maxsmi
import pytest
import sys

from maxsmi.utils_smiles import smi2can, smi2rand, smi2unique_rand


def test_maxsmi_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "maxsmi" in sys.modules


@pytest.mark.parametrize(
    "smiles, solution",
    [
        ('C', 'C'),
        ('OC', 'CO'),
    ],
)
def test_smi2can(smiles, solution):
    can_smi = smi2can(smiles)
    assert solution == can_smi


@pytest.mark.parametrize(
    "smiles, solution",
    [
        ('C', ['C', 'C', 'C']),
    ],
)
def test_smi2rand(smiles, solution):
    rand_smi = smi2rand(smiles, int_aug=3)
    assert solution == rand_smi


@pytest.mark.parametrize(
    "smiles, solution",
    [
        ('C', ['C']),
        ('CO', ['OC', 'CO'])
    ],
)
def test_smi2unique_rand(smiles, solution):
    ran_unique_smi = smi2unique_rand(smiles, int_aug=3)
    assert solution == ran_unique_smi
