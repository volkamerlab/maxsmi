"""
utils.py
Python utils functionalities.

Handles the primary functions
"""

import argparse


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
        raise argparse.ArgumentTypeError("Boolean value expected.")
