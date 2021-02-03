"""
pytorch_data.py
Pytorch data set

Handles the primary functions
"""

from torch.utils.data import Dataset


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
