"""
utils_data.py
Retrieve data from MoleculeNet.
http://moleculenet.ai/

Handles the primary functions
"""

import pandas as pd


def data_retrieval(target_data="ESOL"):
    """
    Retrieve data from MoleculeNet.

    Parameters
    ----------
    target_data: str
        The target data to be considered. The default is the ESOL data set.

    Returns
    -------
    data: pd.Pandas
        Pandas data frame.
    """

    if target_data == "ESOL":
        url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
    elif target_data == "free solv":
        url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv"
    elif target_data == "lipophilicity":
        url = (
            "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"
        )
    else:
        print("Invalid data. Choosing ESOL by default. ")
        url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"

    data = pd.read_csv(url)
    print("Shape of data set: ", data.shape)

    return data