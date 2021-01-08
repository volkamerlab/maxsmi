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

    if target_data == "free solv":
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

    print("Shape of data set: ", df.shape)

    return df


def augmented_data(aug_smiles_df, target_df):
    """
    Transforms augmented lists into augmented data frame.

    Parameters
    ----------
    aug_smiles_df : pd.Pandas
        Data frame containing lists of augmented smiles.
    target_df : pd.Pandas
        Data frame with target values.

    Returns
    -------
    pd.Pandas :
        Data frame with augmented smiles and associated target value.
    """

    augmented_data = []
    for i, (smile, target) in enumerate(zip(aug_smiles_df, target_df)):
        # concatenate by columns two series:
        # 1st: randomized smiles 2. necessary repeted associated target value
        smile_target = pd.concat(
            [pd.DataFrame(smile), pd.DataFrame([target for number in smile])], axis=1
        )
        augmented_data.append(smile_target)
    aug_df = pd.concat(augmented_data, axis=0)
    aug_df.columns = ["smiles", "target"]
    return aug_df