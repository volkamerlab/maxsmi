"""
utils_smi.py
Find the optimal SMILES augmentationn for accurate prediction.

Handles the primary functions
"""

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

    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol  = Chem.MolFromSmiles(smiles)

    if mol is None:
        # print('Faulty molecule in RDKit')
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
    
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol  = Chem.MolFromSmiles(smile)

    if mol is None:
        # print('Faulty molecule in RDKit')
        return None
    else:
        return [Chem.MolToSmiles(mol, canonical=False, doRandom=True) for nb in range(int_aug)]