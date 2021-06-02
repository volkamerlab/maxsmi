"""
utils_data.py
Retrieve data from MoleculeNet.
http://moleculenet.ai/

Handles the primary functions
"""

import logging
import pandas as pd
from maxsmi.utils_smiles import get_num_heavy_atoms


def data_retrieval(target_data="ESOL"):
    """
    Retrieve data from MoleculeNet.

    Parameters
    ----------
    target_data: str
        The target data to be considered. The default is the ESOL data set.

    Returns
    -------
    dataframe: pd.Pandas
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

    elif target_data == "ESOL_small":
        data = process_ESOL()
        task = "measured log solubility in mols per litre"

    else:
        if target_data != "ESOL":
            logging.warning("Invalid data. Choosing ESOL by default.")
        url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
        data = pd.read_csv(url)
        task = "measured log solubility in mols per litre"

    dataframe = data[[task, "smiles"]]
    dataframe = dataframe.rename(columns={task: "target"})

    return dataframe


def process_ESOL(num_heavy_atoms=25):
    """
    Considers ony a subset of the ESOL data, where molecule with up to `num_heavy_atoms` heavy atoms are retained.

    Parameters
    ----------
    num_heavy_atoms : int
        The molecules with `num_heavy_atoms` to keep.

    Return
    ------
    dataframe: pd.Pandas
        Pandas data frame with small molecules only.

    """
    url = (
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
    )
    data = pd.read_csv(url)
    data["num_heavy_atom"] = data["smiles"].apply(get_num_heavy_atoms)

    return data[data["num_heavy_atom"] <= num_heavy_atoms]
