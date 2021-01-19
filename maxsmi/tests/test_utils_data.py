"""
Unit and regression test for the maxsmi package.
"""

# Import package, test suite, and other packages as needed
# import maxsmi
import pytest
import sys
from maxsmi.utils_data import data_retrieval


def test_maxsmi_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "maxsmi" in sys.modules


####################
@pytest.mark.parametrize(
    "task, solution",
    [
        ("ESOL", "OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O "),
        ("lipophilicity", "Cn1c(CN2CCN(CC2)c3ccc(Cl)cc3)nc4ccccc14"),
        ("dncn", "OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O "),
        ("free solv", "CN(C)C(=O)c1ccc(cc1)OC"),
    ],
)
def test_data_retrieval(task, solution):
    dataframe = data_retrieval(task)
    assert solution == dataframe.smiles[0]
