"""
Unit and regression test for the maxsmi package.
"""

# Import package, test suite, and other packages as needed
# import maxsmi
import pytest
import sys

from maxsmi.utils_smiles import (
    smi2can,
    smi2rand,
    smi2max_rand,
    identify_disconnected_structures,
    smi2selfies,
    smi2deepsmiles,
    control_smiles_duplication,
)


def test_maxsmi_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "maxsmi" in sys.modules


####################
@pytest.mark.parametrize(
    "smiles, solution",
    [("C", "C"), ("OC", "CO"), ("KCahsbl", None)],
)
def test_smi2can(smiles, solution):
    can_smi = smi2can(smiles)
    assert solution == can_smi


@pytest.mark.parametrize(
    "smiles, int_aug, solution",
    [("C", 3, ["C", "C", "C"]), ("sakjncal", 3, None), ("OC", 0, ["OC"])],
)
def test_smi2rand(smiles, int_aug, solution):
    rand_smi = smi2rand(smiles, int_aug)
    assert solution == rand_smi


def test_smi2rand_exception():
    with pytest.raises(Exception):
        assert smi2rand("OC", -1)


@pytest.mark.parametrize(
    "smiles, solution",
    [("C.", None), ("CCC", "CCC")],
)
def test_identify_disconnected_structures(smiles, solution):
    disconnected_smi = identify_disconnected_structures(smiles)
    assert solution == disconnected_smi


@pytest.mark.parametrize(
    "smiles, max_duplication, solution",
    [
        ("Csab", 3, None),
        ("CO", -1, ["CO"]),
        ("CCC", 300, ["CCC", "C(C)C"]),
    ],
)
def test_smi2max_rand(smiles, max_duplication, solution):
    ran_max_smi = smi2max_rand(smiles, max_duplication)
    assert solution == ran_max_smi


@pytest.mark.parametrize(
    "smiles, control_function, solution",
    [
        (["CCC", "CCC"], lambda x: 1, ["CCC"]),
        (["CCC", "CCC"], lambda x: x, ["CCC", "CCC"]),
        (["CCC", "CCC", "C(C)C", "C(C)C", "C(C)C"], lambda x: 1, ["CCC", "C(C)C"]),
        (["CCC", "CCC", "C(C)C"], lambda x: x, ["CCC", "CCC", "C(C)C"]),
        (["CCC", "CCC", "C(C)C"], lambda x: x / 2, ["CCC", "C(C)C"]),
    ],
)
def test_control_smiles_duplication(smiles, control_function, solution):
    controlled_duplicates = control_smiles_duplication(
        smiles, duplicate_control=control_function
    )
    assert solution == controlled_duplicates


@pytest.mark.parametrize(
    "smiles, solution", [("c1ccccc1", ["[C][=C][C][=C][C][=C][Ring1][Branch1_2]"])]
)
def test_smi2selfies(smiles, solution):
    selfies = smi2selfies(smiles)
    assert solution == selfies


@pytest.mark.parametrize(
    "smiles, solution", [("c1cccc(C(=O)Cl)c1", ["cccccC=O)Cl))c6"])]
)
def test_smi2deepsmiles(smiles, solution):
    deepsmiles = smi2deepsmiles(smiles)
    assert solution == deepsmiles
