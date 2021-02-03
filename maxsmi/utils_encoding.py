"""
utils_encoding.py
SMILES encoding for machine learning.

Handles the primary functions
"""
import numpy as np


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


def char_replacement(list_smiles):
    """
    Replace the double characters into single character in a list of SMILES string.

    Parameters
    ----------
    list_smiles: list
        list of SMILES string describing a compound.

    Returns
    -------
    list
        list of SMILES with character replacement.
    """
    return [
        smile.replace("Cl", "L")
        .replace("Br", "R")
        .replace("@@", "$")
        .replace("\\", "|")
        for smile in list_smiles
    ]


def get_unique_elements_as_dict(list_):
    """
    Given a list, obtain a dictonary with unique elements as keys and integer as values.

    Parameters
    ----------
    list_: list
        A generic list of strings.

    Returns
    -------
    dict
        Unique elements of the sorted list with assigned integer.
    """
    all_elements = "".join(list_)
    unique_elements = sorted(list(set(all_elements)))
    return {unique_elem: i for i, unique_elem in enumerate(unique_elements)}


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
