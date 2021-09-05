"""
full_workflow_optimal.py

Workflow to run a pytorch machine learning model on a task with the best augmentation stragegy.

"""
import argparse
import logging
import logging.handlers
import warnings
import os
from datetime import datetime
import itertools

import torch
import torch.nn as nn

from maxsmi.utils_data import data_retrieval
from maxsmi.utils_smiles import (
    smiles_to_canonical,
    is_connected,
    ALL_SMILES_DICT,
)
from maxsmi.utils_encoding import char_replacement, get_max_length
from maxsmi.constants import BACTH_SIZE, LEARNING_RATE
from maxsmi.pytorch_models import model_type
from maxsmi.pytorch_data import AugmentSmilesData
from maxsmi.pytorch_training import model_training
from maxsmi.utils_optimal_model import retrieve_optimal_model
from maxsmi.parser_default import NB_EPOCHS


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        dest="task",
        type=str,
        help="data to be used",
    )
    parser.add_argument(
        "--nb-epochs",
        dest="number_epochs",
        type=int,
        help="Number of epochs for training",
        default=NB_EPOCHS,
    )

    args = parser.parse_args()

    if args.task == "chembl":
        args.task = "affinity"

    folder = f"maxsmi/prediction_models/{args.task}"
    os.makedirs(folder, exist_ok=True)

    # Logging information
    log_file_name = "output.log"
    logging.basicConfig(filename=f"{folder}/{log_file_name}", level=logging.INFO)
    logging.info(f"Start at {datetime.now()}")
    logging.info(f"Data and task: {args.task}")

    # ================================
    # Computing device
    # ================================

    is_cuda = torch.cuda.is_available()

    if is_cuda:
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(device)
        logging.info(f"CUDA available: {is_cuda} with {device_name}")
    else:
        device = torch.device("cpu")
        logging.info(f"CUDA available: {is_cuda}")

    time_execution_start = datetime.now()

    # ================================
    # Data
    # ================================

    # Read data
    data = data_retrieval(args.task)

    # Canonical SMILES
    data["canonical_smiles"] = data["smiles"].apply(smiles_to_canonical)
    data = data[data["canonical_smiles"].apply(is_connected)]

    logging.info(f"Shape of data set: {data.shape} ")
    logging.info(f"Number of training points before augmentation: {len(data)} ")

    # ================================
    # Retrieve best model
    # ================================
    (
        ml_model,
        augmentation_strategy,
        augmentation_number,
    ) = retrieve_optimal_model(args.task)

    logging.info(f"Augmentation strategy: {augmentation_strategy.__name__}")
    logging.info(f"Augmentation number: {augmentation_number}")
    logging.info(f"Machine learning model: {ml_model}")

    # ================================
    # String encoding & Augmentation
    # ================================
    time_start_augmenting = datetime.now()

    data["augmented_smiles"] = data["canonical_smiles"].apply(
        augmentation_strategy, args=(augmentation_number,)
    )

    time_end_augmenting = datetime.now()
    time_augmenting = time_end_augmenting - time_start_augmenting
    logging.info(f"Time for augmentation {time_augmenting}")

    # ================================
    # Input processing
    # ================================

    # Replace double symbols
    data["new_smiles"] = data["augmented_smiles"].apply(char_replacement)

    # Retrieve all smiles characters
    smi_dict = ALL_SMILES_DICT
    logging.info(f"Number of unique characters: {len(smi_dict)} ")
    logging.info(f"String dictionary: {smi_dict} ")

    # Obtain longest of all smiles
    max_length_smi = get_max_length(
        list(itertools.chain.from_iterable(data["new_smiles"]))
    )
    logging.info(f"Longest smiles in data set: {max_length_smi} ")

    # ==================================
    # Pytorch data
    # ==================================

    data_pytorch = AugmentSmilesData(data)
    logging.info(f"Number of data points in training set: {len(data_pytorch)} ")

    # Pytorch data loader for mini batches
    train_loader = torch.utils.data.DataLoader(
        data_pytorch, batch_size=BACTH_SIZE, shuffle=True
    )

    # ==================================
    # Machine learning ML
    # ==================================

    (ml_model_name, ml_model) = model_type(ml_model, device, smi_dict, max_length_smi)
    logging.info(f"Summary of ml model: {ml_model} ")

    # Loss function
    loss_function = nn.MSELoss()

    # ==================================
    # ML Training
    # ==================================

    logging.info("========")
    logging.info(f"Training for {args.number_epochs} epochs")
    logging.info("========")
    time_start_training = datetime.now()

    loss_per_epoch = model_training(
        data_loader=train_loader,
        ml_model_name=ml_model_name,
        ml_model=ml_model,
        loss_function=loss_function,
        nb_epochs=args.number_epochs,
        is_cuda=is_cuda,
        len_train_data=len(data_pytorch),
        smiles_dictionary=smi_dict,
        max_length_smiles=max_length_smi,
        device_to_use=device,
        learning_rate=LEARNING_RATE,
    )

    logging.info("Training: over")
    time_end_training = datetime.now()
    time_training = time_end_training - time_start_training
    logging.info(f"Time for model training {time_training}")

    # Save model
    torch.save(ml_model.state_dict(), f"{folder}/model_dict.pth")
    logging.info("Script completed. \n \n")
