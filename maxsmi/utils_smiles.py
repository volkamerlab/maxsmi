"""
utils_smi.py
Find the optimal SMILES augmentation for accurate prediction.

Handles the primary functions
"""

from rdkit import Chem
from rdkit.Chem import AllChem

def smi2can(smiles):
    """
    smi2can takes a SMILES and return its canonical form

    Parameters
    ----------
    smiles: str
        SMILES string describing a compound.

    Returns
    -------
        str: the canonical version of the SMILES
            or None if SMILES is not valid.
    """

    mol  = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None
    else:
        return Chem.MolToSmiles(mol, canonical=True, doRandom=False)


def smi2rand(smiles, int_aug=50):
    """
    smi2rand takes a SMILES (non necessarily canonical)
    and returns n random variations of this SMILES.

    Parameters
    ----------
    smiles: str
        SMILES string describing a compound.
    int_aug: int, Optional, default: 50
        the number of random SMILES generated.

    Returns
    -------
        list: a list of int_aug random SMILES
            or None if the initial SMILES is not valid.
    """

    mol  = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None
    else:
        return [Chem.MolToSmiles(mol, canonical=False, doRandom=True) for nb in range(int_aug)]


def smi2unique_rand(smiles, int_aug=50):
    """
    smi2uniquerand takes a SMILES (not necessarily canonical and
    returns n unique random variations of this SMILES

    Parameters
    ----------
    smiles: str
        SMILES string describing a compound.
    int_aug: int, Optional, default: 50
        the number of random (may not be unique) SMILES generated.

    Returns
    -------
        list: a list of unique random SMILES.
    """

    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol  = Chem.MolFromSmiles(smiles)

    if mol is None:
        print('Faulty molecule in RDKit')
        return None
    else:
        smi_unique = []
        for nb in range(int_aug):
            rand = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
            if rand in smi_unique:
                pass
            else:
                smi_unique.append(rand)

        return smi_unique