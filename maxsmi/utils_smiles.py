"""
utils_smiles.py
SMILES augmentation in RDKit.

Handles the primary functions
"""

from rdkit import Chem


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
                for nb in range(int_aug)
            ]
        else:
            return Chem.MolToSmiles(mol, canonical=False, doRandom=False)


def smi2unique_rand(smiles, int_aug=50):
    """
    smi2uniquerand takes a SMILES (not necessarily canonical) and
    returns `int_aug` unique random variations of this SMILES.

    Parameters
    ----------
    smiles : str
        SMILES string describing a compound.
    int_aug : int, Optional, default: 50
        The number of random (unique) SMILES generated.

    Returns
    -------
    list
        A list of unique random SMILES.
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    else:
        if int_aug > 0:
            smi_unique = []
            for nb in range(int_aug):
                rand = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
                if rand in smi_unique:
                    pass
                else:
                    smi_unique.append(rand)

            return smi_unique
        else:
            return Chem.MolToSmiles(mol, canonical=False, doRandom=False)
