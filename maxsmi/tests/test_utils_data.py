"""
Unit and regression test for the maxsmi package.
"""

# Import package, test suite, and other packages as needed
# import maxsmi
import pytest
import sys
from maxsmi.utils_data import data_retrieval, process_ESOL


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
        ("free_solv", "CN(C)C(=O)c1ccc(cc1)OC"),
        ("ESOL_small", "Cc1occc1C(=O)Nc2ccccc2"),
    ],
)
def test_data_retrieval(task, solution):
    dataframe = data_retrieval(task)
    assert solution == dataframe.smiles[0]


@pytest.mark.parametrize(
    "num_heavy_atoms, solution",
    [
        (25, 15),
    ],
)
def test_process_ESOL(num_heavy_atoms, solution):
    dataframe = process_ESOL(num_heavy_atoms)
    assert solution == dataframe.loc[0, "num_heavy_atom"]
