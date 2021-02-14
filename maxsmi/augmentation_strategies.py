import math
from utils_smiles import (
    smi2rand,
    control_smiles_duplication,
)


def no_augmentation(smiles, augmentation_number=0):
    """
    Takes a SMILES and returns it in a list.

    Parameters
    ----------
    smiles : str
        SMILES string describing a compound.

    Returns
    -------
    list
        A list containing the given SMILES.
    """
    return [smiles]


def augmentation_with_duplication(smiles, augmentation_number):
    """
    Takes a SMILES and returns a list of random SMILES with possible duplicates.

    Parameters
    ----------
    smiles : str
        SMILES string describing a compound.
    augmentation_number : int
        The integer to generate the number of random SMILES.

    Returns
    -------
    list
        A list containing the given number of random SMILES, which might include  duplicated SMILES.
    """
    smiles_list = smi2rand(smiles, augmentation_number)
    return control_smiles_duplication(smiles_list, lambda x: x)


def augmentation_without_duplication(smiles, augmentation_number):
    """
    Takes a SMILES and returns a list of unique random SMILES.

    Parameters
    ----------
    smiles : str
        SMILES string describing a compound.
    augmentation_number : int
        The integer to generate the number of random SMILES.

    Returns
    -------
    list
        A list of unique random SMILES (no duplicates).
    """
    smiles_list = smi2rand(smiles, augmentation_number)
    return control_smiles_duplication(smiles_list, lambda x: 1)


def augmentation_with_reduced_duplication(smiles, augmentation_number):
    """
    Takes a SMILES and returns a list of random SMILES with reduced amount of duplicates.

    Parameters
    ----------
    smiles : str
        SMILES string describing a compound.
    augmentation_number : int
        The integer to generate the number of random SMILES.

    Returns
    -------
    list
        A list of random SMILES with reduced amount of duplicates.
    """
    smiles_list = smi2rand(smiles, augmentation_number)
    return control_smiles_duplication(smiles_list, lambda x: math.sqrt(x))