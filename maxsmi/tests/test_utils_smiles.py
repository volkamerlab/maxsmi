"""
Unit and regression test for the maxsmi package.
"""

# Import package, test suite, and other packages as needed
# import maxsmi
import pytest
import sys
import numpy

from maxsmi.utils.utils_smiles import (
    smiles_to_canonical,
    smiles_to_random,
    smiles_to_max_random,
    is_connected,
    smiles_to_selfies,
    smiles_to_deepsmiles,
    control_smiles_duplication,
    get_num_heavy_atoms,
    validity_check,
    smiles_from_folder_name,
    smiles_to_folder_name,
    smiles_to_morgan_fingerprint,
)


def test_maxsmi_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "maxsmi" in sys.modules


####################
@pytest.mark.parametrize(
    "smiles, solution",
    [("C", "C"), ("OC", "CO"), ("KCahsbl", None)],
)
def test_smiles_to_canonical(smiles, solution):
    canonical_smi = smiles_to_canonical(smiles)
    assert solution == canonical_smi


@pytest.mark.parametrize(
    "smiles, int_aug, solution",
    [("C", 3, ["C", "C", "C"]), ("sakjncal", 3, None), ("OC", 0, ["OC"])],
)
def test_smiles_to_random(smiles, int_aug, solution):
    rand_smi = smiles_to_random(smiles, int_aug)
    assert solution == rand_smi


def test_smiles_to_random_exception():
    with pytest.raises(Exception):
        assert smiles_to_random("OC", -1)


@pytest.mark.parametrize(
    "smiles, solution",
    [("C.", False), ("CCC", True)],
)
def test_is_connected(smiles, solution):
    connected_smi = is_connected(smiles)
    assert solution == connected_smi


@pytest.mark.parametrize(
    "smiles, max_duplication, solution",
    [
        ("Csab", 3, None),
        ("CO", -1, ["CO"]),
        ("CCC", 300, ["CCC", "C(C)C"]),
    ],
)
def test_smiles_to_max_random(smiles, max_duplication, solution):
    ran_max_smi = smiles_to_max_random(smiles, max_duplication)
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
    "smiles, solution", [("c1ccccc1", ["[C][=C][C][=C][C][=C][Ring1][=Branch1]"])]
)
def test_smiles_to_selfies(smiles, solution):
    selfies = smiles_to_selfies(smiles)
    assert solution == selfies


@pytest.mark.parametrize(
    "smiles, solution", [("c1cccc(C(=O)Cl)c1", ["cccccC=O)Cl))c6"])]
)
def test_smiles_to_deepsmiles(smiles, solution):
    deepsmiles = smiles_to_deepsmiles(smiles)
    assert solution == deepsmiles


@pytest.mark.parametrize(
    "smiles, solution",
    [("C", 1), ("OC", 2), ("KCahsbl", None)],
)
def test_get_num_heavy_atoms(smiles, solution):
    num_heavy_atoms = get_num_heavy_atoms(smiles)
    assert solution == num_heavy_atoms


@pytest.mark.parametrize(
    "smiles, solution",
    [
        ("CCC", "CCC"),
    ],
)
def test_validity_check(smiles, solution):
    result = validity_check(smiles)
    assert solution == result


def test_validity_check_exception():
    with pytest.raises(Exception):
        assert validity_check("CC111C")


@pytest.mark.parametrize(
    "smiles, solution",
    [("CCC%2F%5C", "CCC/\\"), ("%2A%2A", "**")],
)
def test_smiles_from_folder_name(smiles, solution):
    new_smiles = smiles_from_folder_name(smiles)
    assert solution == new_smiles


@pytest.mark.parametrize(
    "smiles, solution",
    [
        ("CCC/c\\c", "CCC%2Fc%5Cc"),
    ],
)
def test_smiles_to_folder_name(smiles, solution):
    smiles = smiles_to_folder_name(smiles)
    assert solution == smiles


@pytest.mark.parametrize(
    "smiles, solution",
    [
        ("C", numpy.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    ],
)
def test_smiles_to_morgan_fingerprint(smiles, solution):
    smiles = smiles_to_morgan_fingerprint(smiles, nbits=10)
    assert (solution == smiles).all()
