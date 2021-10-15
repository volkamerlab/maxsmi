"""
From smiles to predictions

"""
import argparse
import logging
import logging.handlers
import pandas
import numpy
import warnings
import os
from datetime import datetime

from maxsmi.utils_data import data_retrieval

from maxsmi.utils_smiles import (
    smiles_to_canonical,
    is_connected,
    smiles_to_morgan_fingerprint,
)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

from maxsmi.pytorch_data import FingerprintData

from maxsmi.constants import TEST_RATIO, RANDOM_SEED, BACTH_SIZE, LEARNING_RATE
from maxsmi.pytorch_evaluation import model_evaluation_fingerprint
from maxsmi.pytorch_training import model_training_fingerprint
from maxsmi.utils_evaluation import evaluation_results

from maxsmi.pytorch_models import FeedForwardNetwork

from maxsmi.parser_default import (
    TASK,
    NB_EPOCHS,
)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        dest="task",
        type=str,
        help="data to be used",
        default=TASK,
    )

    parser.add_argument(
        "--nb-epochs",
        dest="number_epochs",
        type=int,
        help="Number of epochs for training",
        default=NB_EPOCHS,
    )

    args = parser.parse_args()

    folder = f"maxsmi/output/{args.task}_fingerprint"
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

    # ================================
    # Splitting
    # ================================

    # Split data into train/test
    train_data, test_data = train_test_split(
        data,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
    )
    logging.info(f"Number of training points before augmentation: {len(train_data)} ")
    logging.info(f"Number of testing points before augmentation: {len(test_data)} ")

    # ================================
    # String encoding
    # ================================

    train_data["fingerprint"] = train_data["canonical_smiles"].apply(
        smiles_to_morgan_fingerprint,
    )

    test_data["fingerprint"] = test_data["canonical_smiles"].apply(
        smiles_to_morgan_fingerprint,
    )

    # ==================================
    # Pytorch data
    # ==================================

    # Pytorch train set
    train_pytorch = FingerprintData(train_data)
    logging.info(f"Number of data points in training set: {len(train_pytorch)} ")

    # Pytorch data loader for mini batches
    train_loader = torch.utils.data.DataLoader(
        train_pytorch, batch_size=BACTH_SIZE, shuffle=True
    )

    # ==================================
    # Machine learning ML
    # ==================================

    ml_model = FeedForwardNetwork()
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

    loss_per_epoch = model_training_fingerprint(
        data_loader=train_loader,
        ml_model=ml_model,
        loss_function=loss_function,
        nb_epochs=args.number_epochs,
        is_cuda=is_cuda,
        len_train_data=len(train_pytorch),
        device_to_use=device,
        learning_rate=LEARNING_RATE,
    )

    logging.info("Training: over")
    time_end_training = datetime.now()
    time_training = time_end_training - time_start_training
    logging.info(f"Time for model training {time_training}")

    # Save model
    torch.save(ml_model.state_dict(), f"{folder}/model_dict.pth")

    # ================================
    # # Evaluate on train set
    # ================================

    # Pytorch data loader
    train_loader = torch.utils.data.DataLoader(
        train_pytorch, batch_size=1, shuffle=False
    )

    (output_pred_train, output_true_train) = model_evaluation_fingerprint(
        data_loader=train_loader, ml_model=ml_model
    )

    output_pred_train = numpy.array(output_pred_train)
    output_true_train = numpy.array(output_true_train)
    output_pred_train = output_pred_train.reshape(output_pred_train.shape[0])
    output_true_train = output_true_train.reshape(output_true_train.shape[0])

    evaluation_train = evaluation_results(output_pred_train, output_true_train)

    logging.info(f"Train metrics (MSE, RMSE, R2): {evaluation_train}")

    # ================================
    # # Evaluate on test set
    # ================================
    logging.info("========")
    logging.info("Testing")
    logging.info("========")

    time_start_testing = datetime.now()

    test_pytorch = FingerprintData(test_data)

    test_loader = torch.utils.data.DataLoader(test_pytorch, batch_size=1, shuffle=False)

    (output_pred_test, output_true_test) = model_evaluation_fingerprint(
        data_loader=test_loader, ml_model=ml_model
    )
    output_pred_test = numpy.array(output_pred_test)
    output_true_test = numpy.array(output_true_test)
    output_pred_test = output_pred_test.reshape(output_pred_test.shape[0])
    output_true_test = output_true_test.reshape(output_true_test.shape[0])

    evaluation_test = evaluation_results(output_pred_test, output_true_test)

    logging.info(f"Test output dimension {output_true_test.shape}")

    logging.info(f"Test metrics (MSE, RMSE, R2): {evaluation_test}")
    time_end_testing = datetime.now()
    time_testing = time_end_testing - time_start_testing
    logging.info(f"Time for model testing {time_testing}")

    time_execution_end = datetime.now()
    time_execution = time_execution_end - time_execution_start
    logging.info(f"Time for model execution {time_execution}")

    # Save metrics
    results_metrics = pandas.DataFrame(
        data={
            "execution": [time_execution],
            "time_training": [time_training],
            "time_testing": [time_testing],
            "loss": [loss_per_epoch],
            "train": [evaluation_train],
            "test": [evaluation_test],
        }
    )
    results_metrics = results_metrics.to_pickle(f"{folder}/results_metrics.pkl")
    logging.info("Script completed. \n \n")
