"""
From smiles to predictions

"""
import numpy
import os
from utils_data import data_retrieval
from utils_smiles import smi2can
from utils_encoding import (
    char_replacement,
    get_unique_elements_as_dict,
    get_max_length,
    one_hot_encode,
    pad_matrix,
)
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import random_split
from pytorch_models import ConvolutionNetwork


if __name__ == "__main__":

    # ================================
    # Data
    # ================================

    # Read data
    df = data_retrieval("ESOL")

    # Canonical SMILES
    df["canonical_smiles"] = df["smiles"].apply(smi2can)

    # ================================
    # Splitting
    # ================================

    # Split data into train/test
    test_ratio = 0.2
    random_id = 1234

    smiles_train, smiles_test, target_train, target_test = train_test_split(
        df["canonical_smiles"],
        df["target"],
        test_size=test_ratio,
        random_state=random_id,
    )

    # ================================
    # Augmentation
    # ================================

    # TODO

    smiles_aug_train = smiles_train
    smiles_aug_test = smiles_test

    # ================================
    # Input processing
    # ================================

    # Replace double symbols
    new_smi_train = smiles_aug_train.apply(char_replacement)
    new_smi_test = smiles_aug_test.apply(char_replacement)

    # Merge all smiles
    all_smiles = new_smi_train.append(new_smi_test)

    # Obtain dictionary for these smiles
    smi_dict = get_unique_elements_as_dict(all_smiles)
    print("Number of unique characters: ", len(smi_dict))

    # Obtain longest of all smiles
    max_length_smi = get_max_length(all_smiles)
    print("Longest smiles in data set: ", max_length_smi)

    # One-hot encode smiles on train and test
    one_hot_train = new_smi_train.apply(one_hot_encode, dictionary=smi_dict)
    one_hot_test = new_smi_test.apply(one_hot_encode, dictionary=smi_dict)

    # Pad for same shape
    input_train = one_hot_train.apply(pad_matrix, max_pad=max_length_smi)
    input_test = one_hot_test.apply(pad_matrix, max_pad=max_length_smi)

    # ================================
    # Machine learning ML
    # ================================

    # ================================
    # Pytorch data
    # ================================

    # Train set

    output_nn_train = torch.tensor(target_train.values).float()
    output_nn_train = output_nn_train.view(-1, 1)
    print("Shape of output: ", output_nn_train.shape)

    input_nn_train = torch.tensor(list(input_train)).float()
    print("Shape of input: ", input_nn_train.shape)

    train_dataset = TensorDataset(input_nn_train, output_nn_train)

    # Use mini batches
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True
    )

    # Initialize ml model
    ml_model = ConvolutionNetwork(nb_char=len(smi_dict), max_length=max_length_smi)
    print("Summary of ml model: ", ml_model)

    # Loss function
    loss_function = nn.MSELoss()

    # Use optimizer for objective function
    optimizer = optim.SGD(ml_model.parameters(), lr=0.001)

    nb_epochs = 5

    loss_per_epoch = []

    # Train model
    for epoch in range(nb_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):

            # True input/output
            input_true, output_true = data

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            output_pred = ml_model(input_true)
            # Objective
            loss = loss_function(output_pred, output_true)
            # Backward
            loss.backward()
            # Optimization
            optimizer.step()
            # Save loss
            running_loss = +loss.item()

        loss_per_epoch.append(running_loss / len(train_dataset))
        print("Epoch : ", epoch + 1)
        # print("Loss: ", loss_per_epoch)

    # Save model
    os.makedirs("saved_model", exist_ok=True)
    torch.save(ml_model.state_dict(), "saved_model/model_dict.pth")

    # ================================
    # # Evaluate on test set
    # ================================

    # Load model
    ml_model.load_state_dict(torch.load("saved_model/model_dict.pth"))

    # Test set

    output_nn_test = torch.tensor(target_test.values).float()
    output_nn_test = output_nn_test.view(-1, 1)
    input_nn_test = torch.tensor(list(input_test)).float()

    test_dataset = TensorDataset(input_nn_test, output_nn_test)

    # Test model
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=len(test_dataset)
    )
    with torch.no_grad():
        for data in test_loader:
            input_true_test, output_true_test = data
            output_pred_test = ml_model(input_true_test)
            loss_pred = loss_function(output_pred_test, output_true_test)
            print("Loss on test set : ", loss_pred.item())