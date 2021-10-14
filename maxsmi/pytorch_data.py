"""
pytorch_data.py
Pytorch data set

Handles the primary functions
"""

import torch
from torch.utils.data import Dataset

from maxsmi.utils_encoding import one_hot_encode, pad_matrix


class AugmentSmilesData(Dataset):
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
    smiles_dictionary,
    maximum_length,
    machine_learning_model_name,
    device_to_use,
    per_mol=False,
):
    """
    One-hot encodes the SMILES in the batch and associated target value.

    Parameters
    ----------
    smiles : tuple
        The SMILES to be considered.
    target : torch.Tensor
        The true target tensor.
    smiles_ dictionary : dict
        The dictionary of SMILES characters.
    maximum_length : int
        The maximum length of SMILES.
    machine_learning_model_name :
        Name of the machine learning model. It can be either "CON1D", "CONV2D", or "RNN".
    device_to_use : torch.device
        The device to use for model instance, "cpu" or "cuda".
    per_mol : bool, default False
        If the target should be viewed per molecule or per SMILES.

    Returns
    -------
    tuple : (torch.tensor, torch.tensor)
        The one-hot encoded true input tensor and tensor output.
    """

    one_hot = [one_hot_encode(smile, smiles_dictionary) for smile in list(smiles)]
    one_hot_pad = [pad_matrix(ohe, maximum_length) for ohe in one_hot]
    input_true = torch.tensor(one_hot_pad, device=device_to_use).float()

    if machine_learning_model_name == "RNN":
        input_true = torch.transpose(input_true, 1, 2)
    if machine_learning_model_name == "CONV2D":
        input_true = input_true.unsqueeze(1)

    output_true = torch.tensor(target, device=device_to_use).float()

    if per_mol:
        output_true = output_true.view(1)
    else:
        output_true = output_true.view(-1, 1)

    return input_true, output_true


class FingerprintData(Dataset):
    """
    Data set for Pytorch compatibility.

    Attributes
    ----------
    pandas_dataframe : pandas.DataFrame
        A data frame containing at least two columns: fingerprint and target.

    Methods
    -------
    __len__()
        Returns the length of the data frame.

    __getitem__(idx)
        Access the idx element of the data frame.
    """

    def __init__(self, pandas_dataframe):
        super().__init__()
        self.pandas_dataframe = pandas_dataframe
        self.fingerprint = self.pandas_dataframe["fingerprint"]
        self.target = self.pandas_dataframe["target"]

    def __len__(self):
        return len(self.pandas_dataframe)

    def __getitem__(self, idx):
        return self.fingerprint.iloc[idx], self.target.iloc[idx]
