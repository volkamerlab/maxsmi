"""
utils.py
Python utils functionalities.

Handles the primary functions
"""

import argparse

from augmentation_strategies import (
    no_augmentation,
    augmentation_with_duplication,
    augmentation_without_duplication,
    augmentation_with_reduced_duplication,
)


def string_to_bool(string):
    """
    Converts a string to a bool.

    Parameters
    ----------
    string : str
        String.

    Returns
    -------
    bool :
        True or False depending on the string.
    """
    if isinstance(string, bool):
        return string
    if string in ("yes", "True", "true", "1"):
        return True
    elif string in ("no", "False", "false", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Choose between 'yes' or 'no'.")


def augmentation_strategy(string):
    """
    Converts a string to an augmentation strategy.

    Parameters
    ----------
    string : str
        String describing the augmentation strategy to be used.

    Returns
    -------
    function :
        The augmentation strategy.
    """
    if string == "no_augmentation":
        return no_augmentation
    elif string == "augmentation_with_duplication":
        return augmentation_with_duplication
    if string == "augmentation_without_duplication":
        return augmentation_without_duplication
    elif string == "augmentation_with_reduced_duplication":
        return augmentation_with_reduced_duplication
    else:
        raise argparse.ArgumentTypeError(
            "Choose between 'no_augmentation',"
            "'augmentation_with_duplication', 'augmentation_without_duplication"
            "or 'augmentation_with_reduced_duplication'."
        )
