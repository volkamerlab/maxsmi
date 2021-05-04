import math
from maxsmi.utils_smiles import smi2rand, control_smiles_duplication, smi2max_rand


def no_augmentation(smiles, augmentation_number=0):
    """
    Takes a SMILES and returns it in a list.

    Parameters
    ----------
    smiles : str
        SMILES string describing a compound.
    augmentation_number : default 0

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
        A list containing the given number of random SMILES, which might include duplicated SMILES.
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
    Takes a SMILES and returns a list of random SMILES with a reduced amount of duplicates.
    The reduction is square root given the number of duplicates.

    Parameters
    ----------
    smiles : str
        SMILES string describing a compound.
    augmentation_number : int
        The integer to generate the number of random SMILES.

    Returns
    -------
    list
        A list of random SMILES with a reduced amount of duplicates.
    """
    smiles_list = smi2rand(smiles, augmentation_number)
    return control_smiles_duplication(smiles_list, lambda x: math.sqrt(x))


def augmentation_maximum_estimation(smiles, max_duplication=100):
    """
    Returns augmented SMILES with estimated maximum number.

    Parameters
    ----------
    smiles : str
        SMILES string describing a compound.
    max_duplication : int, Optional, default: 100
        The number of concecutive redundant SMILES that have to be generated before stopping augmentation process.

    Returns
    -------
    list
        A list of "estimated" maximum unique random SMILES.
    """
    return smi2max_rand(smiles, max_duplication=100)
