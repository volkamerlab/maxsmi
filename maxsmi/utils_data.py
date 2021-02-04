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
        Pandas data frame with two columns:
            - `smiles`: SMILES encoding of the compounds.
            - `target`: the measured target values.
    """

    if target_data == "free_solv":
        url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv"
        data = pd.read_csv(url)
        task = "expt"

    elif target_data == "lipophilicity":
        url = (
            "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"
        )
        data = pd.read_csv(url)
        task = "exp"

    else:
        if target_data != "ESOL":
            print("Invalid data. Choosing ESOL by default. ")
        url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
        data = pd.read_csv(url)
        task = "measured log solubility in mols per litre"

    df = data[[task, "smiles"]]
    df = df.rename(columns={task: "target"})

    return df
