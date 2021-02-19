"""
pytorch_data.py
Pytorch data set

Handles the primary functions
"""

import torch
from torch.utils.data import Dataset

from utils_encoding import one_hot_encode, pad_matrix


class AugmenteSmilesData(Dataset):
    """
    Data set for Pytorch compatibility.

    Attributes
    ----------
    pandas_dataframe : pandas.DataFrame
        A data frame containing at least two columns: smiles and target.
    index_augmentation : bool
        Whether the data set should be expanded with respect to the smiles multiplicity.

    Methods
    -------
    __len__()
        Returns the length of the data frame.

    __getitem__(idx)
        Access the idx element of the data frame.
    """

    def __init__(self, pandas_dataframe, index_augmentation=True):
        super().__init__()
        self.pandas_dataframe = pandas_dataframe
        if index_augmentation:
            self.pandas_dataframe = self.pandas_dataframe.explode(
                "new_smiles", ignore_index=True
            )
        self.smiles = self.pandas_dataframe["new_smiles"]
        self.target = self.pandas_dataframe["target"]

    def __len__(self):
        return len(self.pandas_dataframe)

    def __getitem__(self, idx):
        return self.smiles.loc[idx], self.target.loc[idx]


def data_to_pytorch_format(
    smiles,
    target,
    dictionary,
    maximum_length,
    machine_learning_model,
    device_to_use,
    per_mol=False,
):
    """
    # TODO
    Parameters
    ----------

    Returns
    -------
    tuple :

    """
    one_hot = [one_hot_encode(smile, dictionary) for smile in list(smiles)]
    one_hot_pad = [pad_matrix(ohe, maximum_length) for ohe in one_hot]
    input_true = torch.tensor(one_hot_pad, device=device_to_use).float()

    if machine_learning_model == "RNN":
        input_true = torch.transpose(input_true, 1, 2)
    if machine_learning_model == "CONV2D":
        input_true = input_true.unsqueeze(1)

    output_true = torch.tensor(target, device=device_to_use).float()

    if per_mol:
        output_true = output_true.view(1)
    else:
        output_true = output_true.view(-1, 1)

    return input_true, output_true
