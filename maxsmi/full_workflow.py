"""
From smiles to predictions

"""

from utils_data import data_retrieval
from utils_smiles import smi2can
from utils_encoding import (
    char_replacement,
    get_unique_elements_as_dict,
    get_max_length,
    one_hot_encode,
    pad_matrix,
)
import torch
from pytorch_models import ConvolutionNetwork
import torch.optim as optim
import torch.nn as nn


if __name__ == "__main__":

    # ================
    # DATA
    # ================

    # Read data
    data = data_retrieval("ESOL")
    ## TODO : check smiles and target columns for each data set
    df = data[["measured log solubility in mols per litre", "smiles"]]
    df = df.rename(columns={"measured log solubility in mols per litre": "ESOL"})

    # ================
    # Input processing
    # ================

    # Canonical SMILES
    df["canonical_smiles"] = df["smiles"].apply(smi2can)

    # Replace double symbols
    df["new_smiles"] = df["canonical_smiles"].apply(char_replacement)

    # Obtain dictionary for this smile
    smi_dict = get_unique_elements_as_dict(df["new_smiles"])
    print("Number of unique characters: ", len(smi_dict))

    # One-hot encode smiles
    df["one_hot_smiles"] = df["new_smiles"].apply(one_hot_encode, dictionary=smi_dict)

    # Obtain longest smiles
    max_length_smi = get_max_length(df["new_smiles"])
    print("Longest smiles in data set: ", max_length_smi)

    # Pad for same shape
    df["padded_ohe"] = df["one_hot_smiles"].apply(pad_matrix, max_pad=max_length_smi)

    # ================
    # Machine learning ML
    # ================

    input_nn = torch.tensor(df["padded_ohe"])
    print("Shape of input: ", input_nn.shape)

    output_nn = torch.tensor(df["ESOL"])
    output_nn = output_nn.view(-1, 1)
    print("Shape of output: ", output_nn.shape)

    ml_model = ConvolutionNetwork(nb_char=len(smi_dict), max_length=max_length_smi)
    print(ml_model)
    out = ml_model(input_nn.float())
    print(out.shape)

    # Loss function
    loss_function = nn.MSELoss()

    # Use optimizer for objective function
    optimizer = optim.SGD(ml_model.parameters(), lr=0.01)

    # in your training loop:
    optimizer.zero_grad()  # zero the gradient buffers
    output = ml_model(input_nn.float())
    loss = loss_function(out, output_nn)
    # loss.backward()
    # optimizer.step()  # Does the update

    # Split data train - test
    # Train model
    # Evaluate on train & test