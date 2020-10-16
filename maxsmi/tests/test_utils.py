"""
Unit and regression test for the maxsmi package.
"""

# Import package, test suite, and other packages as needed
import maxsmi
import pytest
import sys

from maxsmi.utils_smiles import smi2can, smi2rand
# from utils_smiles import smi2can, smi2rand

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