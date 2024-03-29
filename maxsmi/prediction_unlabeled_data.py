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
import numpy
import rdkit
from rdkit.Chem import Draw
import torch

from maxsmi.utils.utils_data import data_retrieval, smiles_in_training, data_checker
from maxsmi.utils.utils_smiles import (
    validity_check,
    smiles_to_canonical,
    smiles_to_folder_name,
    smiles_from_folder_name,
    is_connected,
    ALL_SMILES_DICT,
)
from maxsmi.utils.utils_encoding import char_replacement
from maxsmi.utils.utils_prediction import (
    retrieve_longest_smiles_from_optimal_model,
    unlabeled_smiles_max_length,
    character_check,
    mixture_check,
)

from maxsmi.pytorch_utils.pytorch_models import model_type
from maxsmi.pytorch_utils.pytorch_data import AugmentSmilesData
from maxsmi.pytorch_utils.pytorch_evaluation import out_of_sample_prediction
from maxsmi.utils.utils_optimal_model import retrieve_optimal_model

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
        "--smiles_prediction",
        dest="user_smiles",
        type=str,
        help="SMILES for prediction",
    )

    args = parser.parse_args()

    # Check for URL encoded SMILES
    user_smiles = smiles_from_folder_name(args.user_smiles)

    # SMILES validity check
    validity_check(user_smiles)
    mixture_check(user_smiles)
    character_check(user_smiles)

    # Data checker
    data_checker(args.task)

    # URL encoding for folder name
    url_encoded_user_smiles = smiles_to_folder_name(user_smiles)
    folder = f"maxsmi/user_prediction/{args.task}_{url_encoded_user_smiles}"
    os.makedirs(folder, exist_ok=True)

    # Logging information
    log_file_name = "user_prediction_output.log"
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
    logging.info(f"Shape of training data set before processing: {data.shape} ")

    # Canonical SMILES
    data["canonical_smiles"] = data["smiles"].apply(smiles_to_canonical)
    data = data[data["canonical_smiles"].apply(is_connected)]

    logging.info(f"Shape of training data set after processing: {data.shape} ")

    # Create data frame with unlabeled SMILES
    new_data = pandas.DataFrame(columns=["target", "smiles"])
    new_data.loc[0] = [numpy.nan, user_smiles]

    new_data["smiles_in_training"] = new_data["smiles"].apply(
        smiles_in_training, args=(data,)
    )
    logging.info(
        f"SMILES in training data set: {new_data.loc[0, 'smiles_in_training']} "
    )

    # Canonical SMILES
    new_data["canonical_smiles"] = new_data["smiles"].apply(smiles_to_canonical)

    # ================================
    # String encoding & Augmentation
    # ================================
    time_start_augmenting = datetime.now()

    (
        ml_model,
        augmentation_strategy,
        augmentation_number,
    ) = retrieve_optimal_model(args.task)
    longest_smiles = retrieve_longest_smiles_from_optimal_model(args.task)

    new_data["augmented_smiles"] = new_data["canonical_smiles"].apply(
        augmentation_strategy, args=(augmentation_number,)
    )

    logging.info(f"Augmentation strategy: {augmentation_strategy.__name__}")
    logging.info(f"Augmentation number: {augmentation_number}")

    time_end_augmenting = datetime.now()
    time_augmenting = time_end_augmenting - time_start_augmenting
    logging.info(f"Time for augmentation {time_augmenting}")

    # ================================
    # Input processing
    # ================================

    # Replace double symbols
    new_data["new_smiles"] = new_data["augmented_smiles"].apply(char_replacement)

    # Retrieve SMILES' dictionary
    smi_dict = ALL_SMILES_DICT

    # Obtain longest of all smiles
    max_length_smi = longest_smiles
    # Check that all random SMILES are not longer than the maximum length
    [
        unlabeled_smiles_max_length(smiles, max_length_smi)
        for smiles in new_data["new_smiles"][0]
    ]
    logging.info(f"Longest smiles in training data set: {max_length_smi} ")

    # ==================================
    # Machine learning ML
    # ==================================

    (ml_model_name, ml_model) = model_type(ml_model, device, smi_dict, max_length_smi)
    logging.info(f"Summary of ml model used for the prediction: {ml_model} ")
    file_path = f"maxsmi/prediction_models/{args.task}"
    ml_model.load_state_dict(
        torch.load(f"{file_path}/model_dict.pth", map_location=device)
    )

    # ================================
    # # Evaluation
    # ================================
    logging.info("========")
    logging.info("Evaluation")
    logging.info("========\n\n")

    time_start_testing = datetime.now()

    new_pytorch = AugmentSmilesData(new_data)

    new_loader = torch.utils.data.DataLoader(new_pytorch, batch_size=1, shuffle=False)

    output_prediction = out_of_sample_prediction(
        data_loader=new_loader,
        ml_model_name=ml_model_name,
        ml_model=ml_model,
        smiles_dictionary=smi_dict,
        max_length_smiles=max_length_smi,
        device_to_use=device,
    )

    # Create a data frame with important information:
    # canonical smiles, random smiles, mean prediction and standard deviation
    new_ensemble_learning = new_data.copy()

    for index, row in new_data.iterrows():
        # Obtain prediction for each of the random smiles of a given molecule
        multiple_output = numpy.concatenate(
            [output_prediction[smiles][0] for smiles in row["new_smiles"]]
        )
        new_ensemble_learning.loc[index, "per_smiles_prediction"] = numpy.array2string(
            multiple_output, separator=", "
        )
        logging.info(f"Prediction per random SMILES: \n {multiple_output}")

        # Average the predictions for a given molecule
        prediction_per_mol = numpy.mean(multiple_output)

        # Compute the standard deviation for a given molecule
        std_prediction_per_mol = numpy.std(multiple_output)

        # Add the new values to the data frame:
        new_ensemble_learning.loc[index, "average_prediction"] = prediction_per_mol
        new_ensemble_learning.loc[index, "std_prediction"] = std_prediction_per_mol

        logging.info(f" Prediction: {prediction_per_mol}")
        logging.info(f" Confidence: {std_prediction_per_mol} \n\n")

        # Draw and save 2D image of the molecule
        molecule = rdkit.Chem.MolFromSmiles(new_data.canonical_smiles[index])
        Draw.MolToFile(molecule, f"{folder}/2D_molecule.png")

    new_ensemble_learning = new_ensemble_learning.drop(columns=["target", "new_smiles"])
    new_ensemble_learning = new_ensemble_learning.rename(
        columns={"smiles": "user_smiles"}
    )
    new_ensemble_learning.to_csv(
        f"{folder}/user_prediction_table.csv", index=False, sep=","
    )

    time_end_testing = datetime.now()
    time_testing = time_end_testing - time_start_testing
    logging.info(f"Time for model testing {time_testing}")

    time_execution_end = datetime.now()
    time_execution = time_execution_end - time_execution_start
    logging.info(f"Time for model execution {time_execution}")

    logging.info("Script completed. \n \n")
    print(f"Script completed. Output can be found at {folder}/")
