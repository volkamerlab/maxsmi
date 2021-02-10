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

from utils import string_to_bool
from utils_data import data_retrieval

from utils_smiles import (
    smi2can,
    identify_disconnected_structures,
    smi2unique_rand,
    smi2selfies,
    smi2deepsmiles,
)
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

from pytorch_models import (
    Convolutional1DNetwork,
    Convolutional2DNetwork,
    RecurrentNetwork,
)
from pytorch_data import AugmenteSmilesData

# Constants
TEST_RATIO = 0.2
RANDOM_SEED = 1234
STRING_ENCODING = "smiles"
TRAIN_AUGMENTATION = 0
TEST_AUGMENTATION = 0
BACTH_SIZE = 16
LEARNING_RATE = 0.01
NB_EPOCHS = 2
TASK = "ESOL"
ENSEMBLE_LEARNING = True
AUGMENTATION_STRATEGY = smi2unique_rand
ML_MODEL = "CONV1D"

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
        "--eval-strategy",
        dest="ensemble_learning",
        type=string_to_bool,
        help="ensemble learning used as evaluation strategy",
        default=ENSEMBLE_LEARNING,
    )

    args = parser.parse_args()

    folder = (
        f"maxsmi/output/{args.task}_{args.string_encoding}"
        f"_{args.augmentation_train}_{args.augmentation_test}_{args.machine_learning_model}"
    )
    os.makedirs(folder, exist_ok=True)

    # Logging information
    log_file_name = "output.log"
    logging.basicConfig(filename=f"{folder}/{log_file_name}", level=logging.INFO)
    logging.info(f"Start at {datetime.now()}")
    logging.info(f"Data and task: {args.task}")
    logging.info(f"Augmentation strategy: {args.augmentation_strategy}")
    logging.info(f"Evaluation strategy (ensemble learning): {args.ensemble_learning}")
    logging.info(f"Train augmentation: {args.augmentation_train}")
    logging.info(f"Test augmentation: {args.augmentation_test}")
    logging.info(f"Machine learning model: {args.machine_learning_model}")

    time_execution_start = datetime.now()

    # ================================
    # Data
    # ================================

    # Read data
    data = data_retrieval(args.task)

    # Canonical SMILES
    data["canonical_smiles"] = data["smiles"].apply(smi2can)
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

    # ================================
    # String encoding & Augmentation
    # ================================
    if args.string_encoding == "smiles":
        train_data["augmented_smiles"] = train_data["canonical_smiles"].apply(
            args.augmentation_strategy, args=(args.augmentation_train,)
        )
        test_data["augmented_smiles"] = test_data["canonical_smiles"].apply(
            args.augmentation_strategy, args=(args.augmentation_test,)
        )

    elif args.string_encoding == "selfies":
        train_data["augmented_smiles"] = train_data["canonical_smiles"].apply(
            smi2selfies
        )
        test_data["augmented_smiles"] = test_data["canonical_smiles"].apply(smi2selfies)

    elif args.string_encoding == "deepsmiles":
        train_data["augmented_smiles"] = train_data["canonical_smiles"].apply(
            smi2deepsmiles
        )
        test_data["augmented_smiles"] = test_data["canonical_smiles"].apply(
            smi2deepsmiles
        )

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

    # ================================
    # Machine learning ML
    # ================================

    # ================================
    # Pytorch data
    # ================================

    time_start_training = datetime.now()

    # Pytorch train set
    train_pytorch = AugmenteSmilesData(train_data)

    # Pytorch data loader for mini batches
    train_loader = torch.utils.data.DataLoader(
        train_pytorch, batch_size=BACTH_SIZE, shuffle=True
    )

    # Initialize ml model
    if args.machine_learning_model == "CONV1D":
        ml_model = Convolutional1DNetwork(
            nb_char=len(smi_dict), max_length=max_length_smi
        )
    elif args.machine_learning_model == "CONV2D":
        ml_model = Convolutional2DNetwork(
            nb_char=len(smi_dict), max_length=max_length_smi
        )
    else:
        ml_model = RecurrentNetwork(nb_char=len(smi_dict), max_length=max_length_smi)

    logging.info(f"Summary of ml model: {ml_model} ")

    # Loss function
    loss_function = nn.MSELoss()

    # Use optimizer for objective function
    optimizer = optim.SGD(ml_model.parameters(), lr=LEARNING_RATE)

    nb_epochs = NB_EPOCHS
    loss_per_epoch = []

    logging.info("========")
    logging.info("Training")
    logging.info("========")

    # Train model
    for epoch in range(nb_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):

            # SMILES and target
            smiles, target = data

            one_hot = [one_hot_encode(smi, smi_dict) for smi in list(smiles)]
            one_hot_pad = [pad_matrix(ohe, max_length_smi) for ohe in one_hot]
            input_true = torch.tensor(one_hot_pad).float()

            output_true = torch.tensor(target).float()
            output_true = output_true.view(-1, 1)

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

        loss_per_epoch.append(running_loss / len(train_pytorch))
        if epoch % 10 == 0:
            logging.info(f"Epoch : {epoch + 1} ")

    logging.info("Training: over")
    time_end_training = datetime.now()
    time_training = time_end_training - time_start_training
    logging.info(f"Time for model training {time_training}")

    # ================================
    # # Evaluate on train set
    # ================================

    one_hot = [one_hot_encode(smi, smi_dict) for smi in list(train_pytorch.smiles)]
    one_hot_pad = [pad_matrix(ohe, max_length_smi) for ohe in one_hot]
    input_train = torch.tensor(one_hot_pad).float()

    output_train = torch.tensor(train_pytorch.target).float()
    output_train = output_train.view(-1, 1)

    logging.info(f"Train input dimension: {input_train.shape}")
    logging.info(f"Train output dimension: {output_train.shape}")

    evaluation_train = evaluation_results(output_train, ml_model(input_train))

    logging.info(f"Train metrics: {evaluation_train}")
    # Save model
    torch.save(ml_model.state_dict(), f"{folder}/model_dict.pth")

    # ================================
    # # Evaluate on test set
    # ================================
    logging.info("========")
    logging.info("Testing")
    logging.info("========")

    # Load model
    # ml_model.load_state_dict(torch.load(f"{folder}/model_dict.pth"))

    # Test set

    time_start_testing = datetime.now()

    with torch.no_grad():
        if args.ensemble_learning:
            test_pytorch = AugmenteSmilesData(test_data, index_augmentation=False)
            output_true_test = []
            output_pred_test = []

            for item in test_pytorch.pandas_dataframe.index:

                # Retrive list of random smiles for a given index
                multiple_smiles = test_pytorch.smiles.__getitem__(item)

                one_hot = [one_hot_encode(smi, smi_dict) for smi in multiple_smiles]
                one_hot_pad = [pad_matrix(ohe, max_length_smi) for ohe in one_hot]
                multiple_input = torch.tensor(one_hot_pad).float()

                if len(multiple_input.shape) < 3:
                    multiple_input = multiple_input.reshape(
                        (1, multiple_input.shape[0], multiple_input.shape[1])
                    )

                # Obtain prediction for each of the random smiles of a given molecule
                multiple_output = ml_model(multiple_input)
                # Average the predictions for a given molecule
                prediction_per_mol = torch.mean(multiple_output, dim=0)
                # Retrieve true target of a given molecule
                output_true_test_per_mol = torch.tensor(
                    test_pytorch.target.__getitem__(item)
                )

                output_true_test.append(output_true_test_per_mol)
                output_pred_test.append(prediction_per_mol)

            output_pred_test = torch.tensor(output_pred_test)
            output_true_test = torch.tensor(output_true_test)

        else:
            test_pytorch = AugmenteSmilesData(test_data, index_augmentation=True)
            one_hot = [
                one_hot_encode(smi, smi_dict) for smi in list(test_pytorch.smiles)
            ]
            one_hot_pad = [pad_matrix(ohe, max_length_smi) for ohe in one_hot]
            input_true_test = torch.tensor(one_hot_pad).float()

            output_true_test = torch.tensor(test_pytorch.target).float()
            output_true_test = output_true_test.view(-1, 1)

            output_pred_test = ml_model(input_true_test)

        loss_pred = loss_function(output_pred_test, output_true_test)
        evaluation_test = evaluation_results(output_true_test, output_pred_test)

        logging.info(f"Test output dimension {output_true_test.shape}")

    logging.info(f"Test metrics: {evaluation_test}")
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
