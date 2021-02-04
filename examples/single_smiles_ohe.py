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
    smile = "Nc2nc1n(COCCO)cnc1c(=O)[nH]2"
    print("Length of original smile: ", len(smile))

    # Replace double symbols
    smile = char_replacement(smile)
    print("Length of double characters replaced smile : ", len(smile))

    # Obtain dictionary for this smile
    smi_dict = get_unique_elements_as_dict(smile)
    print("Number of unique characters: ", len(smi_dict))

    # One-hot encode smile
    ohe = one_hot_encode(smile, smi_dict)
    print("Shape of one-hot encoded matrix: ", ohe.shape)
