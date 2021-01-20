"""
From smiles to predictions

"""
import argparse
import logging
import logging.handlers
import pandas
import os
from datetime import datetime
from utils_data import data_retrieval, augmented_data

from utils_smiles import smi2can, identify_disconnected_structures
from utils_smiles import *
from utils_encoding import (
    char_replacement,
    get_unique_elements_as_dict,
    get_max_length,
    one_hot_encode,
    pad_matrix,
)
from utils_evaluation import evaluation_results
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset
from pytorch_models import ConvolutionNetwork

# Constants
TEST_RATIO = 0.2
RANDOM_SEED = 1234
TRAIN_AUGMENTATION = 5
TEST_AUGMENTATION = 2
BACTH_SIZE = 16
LEARNING_RATE = 0.01
NB_EPOCHS = 20
TASK = "ESOL"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        dest="task",
        type=str,
        help="data to be used",
        default="ESOL",
    )
    parser.add_argument(
        "--aug-train",
        dest="augmentation_train",
        type=int,
        help="aug-train will be generated on train set",
        default=TRAIN_AUGMENTATION,
    )
    parser.add_argument(
        "--aug-test",
        dest="augmentation_test",
        type=int,
        help="aug-test will be generated on test set",
        default=TEST_AUGMENTATION,
    )
    parser.add_argument(
        "--aug-strategy",
        dest="augmentation_strategy",
        type=int,
        help="augmentation strategy to be used",
        default=smi2unique_rand,
    )

    args = parser.parse_args()

    folder = (
        f"maxsmi/output/{args.task}_{args.augmentation_train}_{args.augmentation_test}"
    )
    os.makedirs(folder, exist_ok=True)

    # Logging information
    log_file_name = "output.log"
    logging.basicConfig(filename=f"{folder}/{log_file_name}", level=logging.INFO)

    logging.info(f"Data and task: {args.task}")
    logging.info(f"Augmentation strategy: {args.augmentation_strategy}")
    logging.info(f"Train augmentation: {args.augmentation_train}")
    logging.info(f"Test augmentation: {args.augmentation_test}")

    time_execution_start = datetime.now()

    # ================================
    # Data
    # ================================

    time_start_data = datetime.now()

    # Read data
    df = data_retrieval(args.task)

    # Canonical SMILES
    df["canonical_smiles"] = df["smiles"].apply(smi2can)
    df["disconnected_smi"] = df["canonical_smiles"].apply(
        identify_disconnected_structures
    )
    df = df.dropna(axis=0)
    df = df.drop(["disconnected_smi", "smiles"], axis=1)

    logging.info(f"Shape of data set: {df.shape} ")

    # ================================
    # Splitting
    # ================================

    # Split data into train/test
    smiles_train, smiles_test, target_train, target_test = train_test_split(
        df["canonical_smiles"],
        df["target"],
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
    )

    # ================================
    # Augmentation
    # ================================
    smiles_aug_train = smiles_train.apply(
        args.augmentation_strategy, args=(args.augmentation_train,)
    )
    smiles_aug_test = smiles_test.apply(
        args.augmentation_strategy, args=(args.augmentation_test,)
    )

    augmented_train = augmented_data(smiles_aug_train, target_train)
    augmented_test = augmented_data(smiles_aug_test, target_test)

    # ================================
    # Input processing
    # ================================

    # Replace double symbols
    new_smi_train = augmented_train["smiles"].apply(char_replacement)
    new_smi_test = augmented_test["smiles"].apply(char_replacement)

    # Merge all smiles
    all_smiles = new_smi_train.append(new_smi_test)

    # Obtain dictionary for these smiles
    smi_dict = get_unique_elements_as_dict(all_smiles)
    logging.info(f"Number of unique characters: {len(smi_dict)} ")

    # Obtain longest of all smiles
    max_length_smi = get_max_length(all_smiles)
    logging.info(f"Longest smiles in data set: {max_length_smi} ")

    # One-hot encode smiles on train and test
    one_hot_train = new_smi_train.apply(one_hot_encode, dictionary=smi_dict)
    one_hot_test = new_smi_test.apply(one_hot_encode, dictionary=smi_dict)

    # Pad for same shape
    input_train = one_hot_train.apply(pad_matrix, max_pad=max_length_smi)
    input_test = one_hot_test.apply(pad_matrix, max_pad=max_length_smi)

    time_end_data = datetime.now()
    time_data = time_end_data - time_start_data
    logging.info(f"Time for data processing {time_data}")

    # ================================
    # Machine learning ML
    # ================================

    # ================================
    # Pytorch data
    # ================================

    time_start_training = datetime.now()

    # Train set

    output_nn_train = torch.tensor(augmented_train["target"].values).float()
    output_nn_train = output_nn_train.view(-1, 1)
    logging.info(f"Shape of train output: {output_nn_train.shape} ")

    input_nn_train = torch.tensor(list(input_train)).float()
    logging.info(f"Shape of train input: {input_nn_train.shape} ")

    train_dataset = TensorDataset(input_nn_train, output_nn_train)

    # Use mini batches
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BACTH_SIZE, shuffle=True
    )

    # Initialize ml model
    ml_model = ConvolutionNetwork(nb_char=len(smi_dict), max_length=max_length_smi)
    logging.info(f"Summary of ml model: {ml_model} ")

    # Loss function
    loss_function = nn.MSELoss()

    # Use optimizer for objective function
    optimizer = optim.SGD(ml_model.parameters(), lr=LEARNING_RATE)

    nb_epochs = NB_EPOCHS
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
        if epoch % 10 == 0:
            logging.info(f"Epoch : {epoch + 1} ")

    logging.info("Training: over")
    time_end_training = datetime.now()
    time_training = time_end_training - time_start_training
    logging.info(f"Time for model training {time_training}")

    # ================================
    # # Evaluate on train set
    # ================================

    evaluation_train = evaluation_results(output_nn_train, ml_model(input_nn_train))
    logging.info(f"Train metrics: {evaluation_train}")
    # Save model
    torch.save(ml_model.state_dict(), f"{folder}/model_dict.pth")

    # ================================
    # # Evaluate on test set
    # ================================
    logging.info("Test set evaluation")

    # Load model
    # ml_model.load_state_dict(torch.load(f"{folder}/model_dict.pth"))

    # Test set

    time_start_testing = datetime.now()

    output_nn_test = torch.tensor(augmented_test["target"].values).float()
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
            evaluation_test = evaluation_results(
                output_true_test, ml_model(input_true_test)
            )
    logging.info(f"Test metrics: {evaluation_test}")
    time_end_testing = datetime.now()
    time_testing = time_end_testing - time_start_testing
    logging.info(f"Time for model testing {time_testing}")

    time_execution_end = datetime.now()
    time_execution = time_execution_end - time_execution_start
    logging.info(f"Time for model execution {time_execution}")

    # Save metrics to a csv
    results_cv = pandas.DataFrame(
        data={
            "execution": [time_execution],
            "data": [time_data],
            "time_training": [time_training],
            "time_testing": [time_testing],
            "loss": [loss_per_epoch],
            "train": [evaluation_train],
            "test": [evaluation_test],
        }
    )
    results_cv = results_cv.to_csv(f"{folder}/results_metrics.csv")
