"""
utils_smiles.py
SMILES augmentation in RDKit.

Handles the primary functions
"""

import math
from collections import Counter
import itertools
from rdkit import Chem
import selfies
import deepsmiles
from maxsmi.utils_encoding import get_unique_elements_as_dict


def validity_check(smiles):
    """
    Aborts the program if the SMILES is unvalid.

    Parameters
    ----------
    smiles : str
        SMILES string describing a compound.

    Returns
    -------
    str :
        the valid SMILES, or raises an error otherwise.

    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Unvalid SMILES. Program aborting")
    else:
        return smiles


def smiles_to_canonical(smiles):
    """
    Takes a SMILES and return its canonical form.

    Parameters
    ----------
    smiles : str
        SMILES string describing a compound.

    Returns
    -------
    str
        The canonical version of the SMILES or None if SMILES is not valid.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None
    else:
        return Chem.MolToSmiles(mol, canonical=True, doRandom=False)


def is_connected(smiles):
    """
    Identifiy connected SMILES through the dot symbol.

    Parameters
    ----------
    smiles : str
        SMILES string describing a compound.

    Returns
    -------
    bool :
        True if the SMILES is connected. False otherwise.

    """
    if "." in smiles:
        return False
    else:
        return True


def smiles_to_random(smiles, int_aug=50):
    """
    Takes a SMILES (not necessarily canonical) and returns `int_aug` random variations of this SMILES.

    Parameters
    ----------
    smiles : str
        SMILES string describing a compound.
    int_aug : int, Optional, default: 50
        The number of random SMILES generated.

    Returns
    -------
    list
        A list of `int_aug` random (may not be unique) SMILES or None if the initial SMILES is not valid.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None
    else:
        if int_aug > 0:
            return [
                Chem.MolToSmiles(mol, canonical=False, doRandom=True)
                for _ in range(int_aug)
            ]
        elif int_aug == 0:
            return [smiles]
        else:
            raise ValueError("int_aug must be greater or equal to zero.")


def smiles_to_max_random(smiles, max_duplication=10):
    """
    Returns estimated maximum number of random SMILES.

    Parameters
    ----------
    smiles : str
        SMILES string describing a compound.
    max_duplication : int, Optional, default: 10
        The number of consecutive redundant SMILES that have to be generated before stopping augmentation process.

    Returns
    -------
    list
        A list of "estimated" maximum unique random SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    else:
        if max_duplication > 0:
            smi_unique = []
            counter = 0
            while counter < max_duplication:
                rand = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
                if rand not in smi_unique:
                    smi_unique.append(rand)
                    counter = 0
                else:
                    counter += 1
            return smi_unique
        else:
            return [Chem.MolToSmiles(mol, canonical=False, doRandom=False)]


def control_smiles_duplication(random_smiles, duplicate_control=lambda x: 1):
    """
    Returns augmented SMILES with the number of duplicates controlled by the function duplicate_control.

    Parameters
    ----------
    random_smiles : list
        A list of random SMILES, can be obtained by `smiles_to_random()`.
    duplicate_control : func, Optional, default: 1
        The number of times a SMILES will be duplicated, as function of the number of times
        it was included in `random_smiles`.
        This number is rounded up to the nearest integer.

    Returns
    -------
    list
        A list of random SMILES with duplicates.

    Notes
    -----
    When `duplicate_control=lambda x: 1`, then the returned list contains only unique SMILES.
    """
    counted_smiles = Counter(random_smiles)
    smiles_duplication = {
        smiles: math.ceil(duplicate_control(counted_smiles[smiles]))
        for smiles in counted_smiles
    }
    return list(
        itertools.chain.from_iterable(
            [[smiles] * smiles_duplication[smiles] for smiles in smiles_duplication]
        )
    )


def smiles_to_selfies(smiles):
    """
    Takes a SMILES and return the selfies encoding.

    Parameters
    ----------
    smiles : str
        SMILES string describing a compound.

    Returns
    -------
    list
        A list of SELFIES encoding of the molecule.
    """

    return [selfies.encoder(smiles)]


def smiles_to_deepsmiles(smiles):
    """
    Takes a SMILES and return the DeepSMILES encoding.

    Parameters
    ----------
    smiles : str
        SMILES string describing a compound.

    Returns
    -------
    list
        A list of DeepSmiles encoding of the molecule.
    """
    converter = deepsmiles.Converter(rings=True, branches=True)
    return [converter.encode(smiles)]


def get_num_heavy_atoms(smiles):
    """
    Takes a SMILES and return the number of heavy atoms of the molecule.

    Parameters
    ----------
    smiles : str
        SMILES string describing a compound.

    Returns
    -------
    int
        The number of heavy atoms of the molecule.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None
    else:
        return mol.GetNumHeavyAtoms()


ALL_SMILES_CHARACTERS = [
    # See https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html
    # Bonds
    "-",  # single bond
    "=",  # double bond
    "#",  # triple bond
    "/",  # directional bond
    "\\",  # directional bond
    # Chirality
    "@",  # chirality specification
    "$",  # replacement for @@
    # Formal charge
    "+",  # formal charge
    "-",  # formal charge
    # Branches
    "(",  # branch opening
    ")",  # branch closing
    # Rings
    "0",  # ring opening and closure
    "1",  #
    "2",  #
    "3",  #
    "4",  #
    "5",  #
    "6",  #
    "7",  #
    "8",  #
    "9",  # ring opening and closure
    "%",  # ring nb > 9
    # Atoms
    "B",  # Boron
    "C",  # Carbon
    "E",  # replacement for Se
    "F",  # Fluorine
    "H",  # Hydrogen
    "I",  # Iodine
    "K",  # Potassium
    "L",  # replacement for Cl
    "N",  # Nitrogen
    "O",  # Oxygen
    "P",  # Phosphorus
    "R",  # replacement for Br
    "S",  # Sulfur
    "T",  # replacement for Si
    "Z",  # replacement for Zn
    # aromatic atoms
    "b",  # boron
    "c",  # carbon
    "e",  # remplacement for se
    "i",  # iodine
    "n",  # nitrogen
    "o",  # oxygen
    "s",  # sulfur
    # Extra
    ".",  # disconnected structure
    "*",  # wild card
    ":",  # atom map (reaction)
    "[",  # for non organic or unormal valence
    "]",  # same
]

ALL_SMILES_DICT = get_unique_elements_as_dict(ALL_SMILES_CHARACTERS)
