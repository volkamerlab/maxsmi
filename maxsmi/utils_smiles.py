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


def smi2can(smiles):
    """
    smi2can takes a SMILES and return its canonical form.

    Parameters
    ----------
    smiles : str
        SMILES string describing a compound.

    Returns
    -------
    str
        The canonical version of the SMILES
        or None if SMILES is not valid.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None
    else:
        return Chem.MolToSmiles(mol, canonical=True, doRandom=False)


def identify_disconnected_structures(smiles):
    """
    Identifiy disconnected structure through the dot symbol.

    Parameters
    ----------
    smiles : str
        SMILES string describing a compound.

    Returns
    -------
    str :
        the SMILES if it's not disconnected. None otherwise.

    """
    if "." in smiles:
        return None
    else:
        return smiles


def smi2rand(smiles, int_aug=50):
    """
    smi2rand takes a SMILES (not necessarily canonical)
    and returns `int_aug` random variations of this SMILES.

    Parameters
    ----------
    smiles : str
        SMILES string describing a compound.
    int_aug : int, Optional, default: 50
        The number of random SMILES generated.

    Returns
    -------
    list
        A list of `int_aug` random (may not be unique) SMILES
        or None if the initial SMILES is not valid.
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
            raise Exception("int_aug must be greater or equal to zero.")


def smi2max_rand(smiles, max_duplication=10):
    """
    Returns augmented SMILES with estimated maximum number.

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
    Returns augmented SMILES with the number of duplicates controlled
    by the function duplicate_control.

    Parameters
    ----------
    random_smiles : list
        A list of random SMILES, can be obtained by smi2rand.
    duplicate_control : func, Optional, default: 1
        The number of times a SMILES will be duplicated,
        as function of the number of times it was included in random_smiles.
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


def smi2selfies(smiles):
    """
    smi2selfies takes a SMILES and return the selfies encoding.

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


def smi2deepsmiles(smiles):
    """
    smi2deepsmiles takes a SMILES and return the DeepSMILES encoding.

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
