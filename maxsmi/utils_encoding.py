"""
utils_encoding.py
SMILES encoding for machine learning.

Handles the primary functions
"""
import numpy as np
from collections import OrderedDict


def get_max_length(list_):
    """
    Computes the longest element in a given list.

    Parameters
    ----------
    list_: list
        A generic list of strings.

    Returns
    -------
    int
        The longest element in the list.
    """

    length_elements = [len(element) for element in list_]
    return max(length_elements)


def char_replacement(smiles):
    """
    Replace the double characters into single character in a SMILES string.

    Parameters
    ----------
    smiles: str
        SMILES string describing a compound.

    Returns
    -------
    str
        SMILES with character replacement.
    """

    smiles = smiles.replace("Cl", "L")
    smiles = smiles.replace("Br", "R")
    smiles = smiles.replace("@@", "$")
    smiles = smiles.replace("\\", "|")

    return smiles


def get_unique_elements_as_dict(list_):
    """
    Given a list, obtain ordered dictonary with unique elements as keys and integer as values.

    Parameters
    ----------
    list_: list
        A generic list of strings.

    Returns
    -------
    ord dict
        Unique elements of the list with assigned integer.
    """
    all_elements = "".join(list_)
    unique_elements = list(set(all_elements))
    dict_ = {unique_elem: i for i, unique_elem in enumerate(unique_elements)}
    return OrderedDict(sorted(dict_.items()))


def one_hot_encode(sequence, dictionary):
    """
    Creates the one-hot encoding of a sequence given a dictionary.

    Parameters
    ----------
    sequence: str
        A sequence of charaters.
    dictionary: dict
        A dictionary which comprises of characters.

    Returns
    -------
    np.array
        The binary matrix of shape `(len(dictionary), len(sequence))`,
        the one-hot encoding of the sequence.

    """
    ohe_matrix = np.zeros((len(dictionary), len(sequence)))
    for i, character in enumerate(sequence):
        ohe_matrix[dictionary[character], i] = 1
    return ohe_matrix


def pad_matrix(matrix, max_pad):
    """
    Pad matrix with zeros for shape dimension.

    Parameters
    ----------
    matrix: np.array
        The one-hot encoded matrix of a smiles.
    max_pad: int
        Dimension to which the padding should be done with zeros.

    Returns
    -------
    np.array
        The padded matrix of shape `(matrix.shape[0], max_pad)`.
    """
    if max_pad < matrix.shape[1]:
        return matrix
    else:
        return np.pad(matrix, ((0, 0), (0, max_pad - matrix.shape[1])))


# def integer_encode(sequence):
#     pass
# def pad_integer_encode(integer):
#     pass
