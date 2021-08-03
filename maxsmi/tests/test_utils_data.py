"""
Unit and regression test for the maxsmi package.
"""

# Import package, test suite, and other packages as needed
# import maxsmi
import pytest
import sys
import pandas
from maxsmi.utils_data import (
    data_retrieval,
    process_ESOL,
    process_ChEMBL,
    smiles_in_training,
    data_checker,
)


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
        ("chembl28", "Brc1cccc(Nc2ncnc3cc4ccccc4cc23)c1"),
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


@pytest.mark.parametrize(
    "uniprotID, solution",
    [
        ("P00533", 11.522878745280336),
    ],
)
def test_process_ChEMBL(uniprotID, solution):
    dataframe = process_ChEMBL(uniprotID)
    assert solution == dataframe.loc[0, "activities.standard_value"]


@pytest.mark.parametrize(
    "smiles, data, solution",
    [
        (
            "CC1CC1",
            pandas.DataFrame(["CCC1CC1", "CC1CC1"], columns=["canonical_smiles"]),
            True,
        ),
        (
            "CC1CC1",
            pandas.DataFrame(["CCC1CC1", "CC1CC1CC"], columns=["canonical_smiles"]),
            False,
        ),
    ],
)
def test_smiles_in_training(smiles, data, solution):
    in_training = smiles_in_training(smiles, data)
    assert solution == in_training


@pytest.mark.parametrize(
    "task, solution",
    [
        ("free_solv", "free_solv"),
        ("ESOL", "ESOL"),
        ("affinity", "affinity"),
        ("lipophilicity", "lipophilicity"),
    ],
)
def test_data_checker(task, solution):
    name = data_checker(task)
    assert solution == name


def test_data_checker_exception():
    with pytest.raises(NameError):
        assert data_checker("lipholicity")
