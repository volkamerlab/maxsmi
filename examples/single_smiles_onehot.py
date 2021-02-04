"""
One-hot encode SMILES

Example with a single SMILES
"""

from maxsmi.utils_encoding import (
    char_replacement,
    get_unique_elements_as_dict,
    one_hot_encode,
)


if __name__ == "__main__":

    # ==================== Example with a SMILES ================
    smiles = "Clc1cc(Cl)c(c(Cl)c1)c2c(Cl)cccc2Cl"
    print("Length of original smile: ", len(smiles))

    # Replace double symbols
    new_smiles = char_replacement([smiles])[0]
    print("Length of double characters replaced smiles : ", len(new_smiles))

    # Obtain dictionary for this smile
    smi_dict = get_unique_elements_as_dict(new_smiles)
    print("Number of unique characters: ", len(smi_dict))

    # One-hot encode smile
    one_hot_smiles = one_hot_encode(new_smiles, smi_dict)
    print("Shape of one-hot encoded matrix: ", one_hot_smiles.shape)
