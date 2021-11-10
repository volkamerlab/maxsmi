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
from sklearn.ensemble import RandomForestRegressor

from maxsmi.utils_data import data_retrieval

from maxsmi.utils_smiles import (
    smiles_to_canonical,
    is_connected,
    smiles_to_morgan_fingerprint,
)
from sklearn.model_selection import train_test_split


from maxsmi.constants import TEST_RATIO, RANDOM_SEED
from maxsmi.utils_evaluation import evaluation_results

from maxsmi.parser_default import TASK

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

    args = parser.parse_args()

    folder = f"maxsmi/output/{args.task}_fingerprint"
    os.makedirs(folder, exist_ok=True)

    # Logging information
    log_file_name = "output.log"
    logging.basicConfig(filename=f"{folder}/{log_file_name}", level=logging.INFO)
    logging.info(f"Start at {datetime.now()}")
    logging.info(f"Data and task: {args.task}")

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
    logging.info(f"Number of training points: {len(train_data)} ")
    logging.info(f"Number of testing points: {len(test_data)} ")

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
    # Machine learning ML
    # ==================================

    # Instantiate model with 100 decision trees
    random_forest = RandomForestRegressor(random_state=42)
    logging.info("Summary of ml model: Random Forest.")

    # ==================================
    # ML Training
    # ==================================
    time_start_training = datetime.now()

    # Train the model on training data
    X_train = numpy.array(list(train_data["fingerprint"]))
    y_train = numpy.array(train_data["target"])
    random_forest.fit(X_train, y_train)

    logging.info("Training: over")
    time_end_training = datetime.now()
    time_training = time_end_training - time_start_training
    logging.info(f"Time for model training {time_training}")

    # ================================
    # # Evaluate on train set
    # ================================
    output_pred_train = random_forest.predict(X_train)
    output_true_train = y_train

    evaluation_train = evaluation_results(output_pred_train, output_true_train)

    logging.info(f"Train metrics (MSE, RMSE, R2): {evaluation_train}")

    # ================================
    # # Evaluate on test set
    # ================================
    logging.info("========")
    logging.info("Testing")
    logging.info("========")

    time_start_testing = datetime.now()

    X_test = numpy.array(list(test_data["fingerprint"]))
    y_test = numpy.array(test_data["target"])
    output_pred_test = random_forest.predict(X_test)
    output_true_test = y_test

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
            "train": [evaluation_train],
            "test": [evaluation_test],
        }
    )
    results_metrics = results_metrics.to_pickle(f"{folder}/results_metrics.pkl")
    logging.info("Script completed. \n \n")
