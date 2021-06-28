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
    Retrieve data from MoleculeNet, or ChEMBL.

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

    elif target_data == "chembl28" or "affinity":
        data = process_ChEMBL()
        task = "activities.standard_value"

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
    dataframe = pd.read_csv(url)
    dataframe["num_heavy_atom"] = dataframe["smiles"].apply(get_num_heavy_atoms)
    dataframe = dataframe.loc[dataframe["num_heavy_atom"] <= num_heavy_atoms]

    return dataframe.reset_index(drop=True)


def process_ChEMBL(uniprotID="P00533"):
    """
    Retrieves pIC50 values in [nM] from ChEMBL v.28.

    Parameters
    ----------
    uniprotID : str, default P00533
        The kinase for which pIC50 measurements are considered. Default is EGFR kinase.

    Returns
    -------
    dataframe: pd.Pandas
        Pandas data frame pIC50 measurements tested against chosen uniprot ID.

    Notes
    -----
    The first curation is done in openkinome, see https://github.com/openkinome/kinodata/releases/tag/v0.2.
    """
    url = "https://github.com/openkinome/kinodata/releases/download/v0.2/activities-chembl28_v0.2.zip"
    dataframe = pd.read_csv(url)
    dataframe = dataframe.dropna(
        subset=[
            "compound_structures.canonical_smiles",
            "component_sequences.sequence",
            "activities.standard_type",
        ]
    )
    dataframe = dataframe[dataframe["activities.standard_type"] == "pIC50"]
    dataframe = dataframe[dataframe["UniprotID"] == uniprotID]
    dataframe = dataframe[
        ["compound_structures.canonical_smiles", "activities.standard_value"]
    ]
    dataframe = dataframe.rename(
        columns={"compound_structures.canonical_smiles": "smiles"}
    )
    return dataframe.reset_index(drop=True)


def smiles_in_training(smiles, data):
    """
    Determines if a SMILES is a dataset.

    Parameters
    ----------
    smiles : str
        SMILES string describing a compound.
    dataframe : pd.Pandas
        A pandas dataframe with a "smiles" column.

    Returns
    -------
    bool :
        If the SMILES is in the dataset.
    """
    if smiles in list(data["smiles"]):
        return True
    else:
        return False
