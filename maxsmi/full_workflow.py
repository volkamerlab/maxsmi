"""
From smiles to predictions

"""
import argparse
import logging
import logging.handlers
import pandas
import warnings
import os
from datetime import datetime
import itertools
import numpy

from maxsmi.utils import string_to_bool, augmentation_strategy
from maxsmi.utils_data import data_retrieval

from maxsmi.utils_smiles import (
    smiles_to_canonical,
    identify_disconnected_structures,
    smiles_to_selfies,
    smiles_to_deepsmiles,
)
from maxsmi.utils_encoding import (
    char_replacement,
    get_unique_elements_as_dict,
    get_max_length,
)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

from maxsmi.pytorch_models import model_type
from maxsmi.pytorch_data import AugmentSmilesData

from maxsmi.constants import TEST_RATIO, RANDOM_SEED, BACTH_SIZE, LEARNING_RATE
from maxsmi.pytorch_evaluation import model_evaluation
from maxsmi.pytorch_training import model_training
from maxsmi.utils_evaluation import evaluation_results

from maxsmi.parser_default import (
    TASK,
    STRING_ENCODING,
    TRAIN_AUGMENTATION,
    TEST_AUGMENTATION,
    ENSEMBLE_LEARNING,
    AUGMENTATION_STRATEGY,
    ML_MODEL,
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
        "--string-encoding",
        dest="string_encoding",
        type=str,
        help="the string encoding for the molecules",
        default=STRING_ENCODING,
    )
    parser.add_argument(
        "--aug-nb-train",
        dest="augmentation_number_train",
        type=int,
        help="aug-train will be generated on train set",
        default=TRAIN_AUGMENTATION,
    )
    parser.add_argument(
        "--aug-nb-test",
        dest="augmentation_number_test",
        type=int,
        help="aug-test will be generated on test set",
        default=TEST_AUGMENTATION,
    )
    parser.add_argument(
        "--aug-strategy-train",
        dest="augmentation_strategy_train",
        type=augmentation_strategy,
        help="augmentation strategy to be used on the train set",
        default=AUGMENTATION_STRATEGY,
    )
    parser.add_argument(
        "--aug-strategy-test",
        dest="augmentation_strategy_test",
        type=augmentation_strategy,
        help="augmentation strategy to be used on the test set",
        default=AUGMENTATION_STRATEGY,
    )
    parser.add_argument(
        "--ml-model",
        dest="machine_learning_model",
        type=str,
        help="machine learning model used for training and testing",
        default=ML_MODEL,
    )
    parser.add_argument(
        "--nb-epochs",
        dest="number_epochs",
        type=int,
        help="Number of epochs for training",
        default=NB_EPOCHS,
    )
    parser.add_argument(
        "--eval-strategy",
        dest="ensemble_learning",
        type=string_to_bool,
        help="ensemble learning used as evaluation strategy",
        default=ENSEMBLE_LEARNING,
    )

    args = parser.parse_args()

    if args.augmentation_strategy_train.__name__ == "no_augmentation":
        args.augmentation_number_train = 0
    if args.augmentation_strategy_test.__name__ == "no_augmentation":
        args.augmentation_number_test = 0
        args.ensemble_learning = False

    folder = (
        f"maxsmi/output/{args.task}_{args.string_encoding}_{args.augmentation_strategy_train.__name__}"
        f"_{args.augmentation_number_train}_{args.augmentation_strategy_test.__name__}"
        f"_{args.augmentation_number_test}_{args.machine_learning_model}"
    )
    os.makedirs(folder, exist_ok=True)

    # Logging information
    log_file_name = "output.log"
    logging.basicConfig(filename=f"{folder}/{log_file_name}", level=logging.INFO)
    logging.info(f"Start at {datetime.now()}")
    logging.info(f"Data and task: {args.task}")
    logging.info(
        f"Augmentation strategy on train: {args.augmentation_strategy_train.__name__}"
    )
    logging.info(
        f"Augmentation strategy on test: {args.augmentation_strategy_test.__name__}"
    )
    logging.info(f"Evaluation strategy (ensemble learning): {args.ensemble_learning}")
    logging.info(f"Train augmentation: {args.augmentation_number_train}")
    logging.info(f"Test augmentation: {args.augmentation_number_test}")
    logging.info(f"Machine learning model: {args.machine_learning_model}")

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
    data["disconnected_smi"] = data["canonical_smiles"].apply(
        identify_disconnected_structures
    )
    data = data.dropna(axis=0)
    data = data.drop(["disconnected_smi", "smiles"], axis=1)

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
    # String encoding & Augmentation
    # ================================
    time_start_augmenting = datetime.now()

    if args.string_encoding == "smiles":
        train_data["augmented_smiles"] = train_data["canonical_smiles"].apply(
            args.augmentation_strategy_train, args=(args.augmentation_number_train,)
        )
        test_data["augmented_smiles"] = test_data["canonical_smiles"].apply(
            args.augmentation_strategy_test, args=(args.augmentation_number_test,)
        )

    elif args.string_encoding == "selfies":
        train_data["augmented_smiles"] = train_data["canonical_smiles"].apply(
            smiles_to_selfies
        )
        test_data["augmented_smiles"] = test_data["canonical_smiles"].apply(
            smiles_to_selfies
        )

    elif args.string_encoding == "deepsmiles":
        train_data["augmented_smiles"] = train_data["canonical_smiles"].apply(
            smiles_to_deepsmiles
        )
        test_data["augmented_smiles"] = test_data["canonical_smiles"].apply(
            smiles_to_deepsmiles
        )

    time_end_augmenting = datetime.now()
    time_augmenting = time_end_augmenting - time_start_augmenting
    logging.info(f"Time for augmentation {time_augmenting}")

    # ================================
    # Input processing
    # ================================

    # Replace double symbols
    train_data["new_smiles"] = train_data["augmented_smiles"].apply(char_replacement)
    test_data["new_smiles"] = test_data["augmented_smiles"].apply(char_replacement)

    # Merge all smiles
    all_smiles = list(
        itertools.chain.from_iterable(
            test_data["new_smiles"].append(train_data["new_smiles"])
        )
    )
    # Obtain dictionary for these smiles
    smi_dict = get_unique_elements_as_dict(list(all_smiles))
    logging.info(f"Number of unique characters: {len(smi_dict)} ")
    logging.info(f"String dictionary: {smi_dict} ")

    # Obtain longest of all smiles
    max_length_smi = get_max_length(all_smiles)
    logging.info(f"Longest smiles in data set: {max_length_smi} ")

    # ==================================
    # Pytorch data
    # ==================================

    # Pytorch train set
    train_pytorch = AugmentSmilesData(train_data)
    logging.info(f"Number of data points in training set: {len(train_pytorch)} ")

    # Pytorch data loader for mini batches
    train_loader = torch.utils.data.DataLoader(
        train_pytorch, batch_size=BACTH_SIZE, shuffle=True
    )

    # ==================================
    # Machine learning ML
    # ==================================

    (ml_model_name, ml_model) = model_type(
        args.machine_learning_model, device, smi_dict, max_length_smi
    )
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
        len_train_data=len(train_pytorch),
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

    # ================================
    # # Evaluate on train set
    # ================================

    # Pytorch data loader
    train_loader = torch.utils.data.DataLoader(
        train_pytorch, batch_size=1, shuffle=False
    )

    (output_pred_train, output_true_train) = model_evaluation(
        data_loader=train_loader,
        ml_model_name=ml_model_name,
        ml_model=ml_model,
        smiles_dictionary=smi_dict,
        max_length_smiles=max_length_smi,
        device_to_use=device,
    )

    all_output_pred_train = numpy.concatenate(
        [output_pred_train[smiles] for new_smiles in train_data["new_smiles"] for smiles in new_smiles]
    )
    all_output_true_train = numpy.concatenate(
        [output_true_train[smiles] for new_smiles in train_data["new_smiles"] for smiles in new_smiles]
    )

    evaluation_train = evaluation_results(all_output_true_train, all_output_pred_train)

    logging.info(f"Train metrics (MSE, RMSE, R2): {evaluation_train}")

    # ================================
    # # Evaluate on test set
    # ================================
    logging.info("========")
    logging.info("Testing")
    logging.info("========")

    time_start_testing = datetime.now()

    test_pytorch = AugmentSmilesData(test_data)

    test_loader = torch.utils.data.DataLoader(test_pytorch, batch_size=1, shuffle=False)

    (output_pred_test, output_true_test) = model_evaluation(
        data_loader=test_loader,
        ml_model_name=ml_model_name,
        ml_model=ml_model,
        smiles_dictionary=smi_dict,
        max_length_smiles=max_length_smi,
        device_to_use=device,
    )

    if args.ensemble_learning:
        # Create a data frame with important information:
        # True value, canonical smiles, random smiles, mean prediction and standard deviation
        test_ensemble_learning = test_data.copy()

        all_output_true_test = []
        all_output_pred_test = []

        for index, row in test_data.iterrows():
            # Obtain prediction for each of the random smiles of a given molecule
            multiple_output = numpy.concatenate(
                [output_pred_test[smiles] for smiles in row["new_smiles"]]
            )
            # Average the predictions for a given molecule
            prediction_per_mol = numpy.mean(multiple_output)

            # Compute the standard deviation for a given molecule
            std_prediction_per_mol = numpy.std(multiple_output)

            # Add the new values to the data frame:
            test_ensemble_learning.loc[index, "average_prediction"] = prediction_per_mol

            test_ensemble_learning.loc[index, "std_prediction"] = std_prediction_per_mol

            all_output_true_test.append(row["target"])
            all_output_pred_test.append(prediction_per_mol)

        test_ensemble_learning.to_pickle(f"{folder}/results_ensemble_learning.pkl")
        all_output_true_test = numpy.array(all_output_true_test)
        all_output_pred_test = numpy.array(all_output_pred_test)
    else:
        all_output_true_test = numpy.concatenate(
            [output_true_test[smiles] for new_smiles in test_data["new_smiles"] for smiles in new_smiles]
        )
        all_output_pred_test = numpy.concatenate(
            [output_pred_test[smiles] for new_smiles in test_data["new_smiles"] for smiles in new_smiles]
        )

    evaluation_test = evaluation_results(all_output_true_test, all_output_pred_test)

    logging.info(f"Test output dimension {all_output_true_test.shape}")

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
