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

from maxsmi.utils_data import data_retrieval, smiles_in_training
from maxsmi.utils_smiles import smiles_to_canonical, ALL_SMILES_CHARACTERS
from maxsmi.utils_encoding import char_replacement
from maxsmi.utils_prediction import retrieve_optimal_model

from maxsmi.pytorch_models import model_type
from maxsmi.pytorch_data import AugmentSmilesData
from maxsmi.pytorch_evaluation import out_of_sample_prediction

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
        dest="new_smiles",
        type=str,
        help="SMILES for prediction",
    )

    args = parser.parse_args()

    folder = f"maxsmi/prediction/{args.task}"
    os.makedirs(folder, exist_ok=True)

    # Logging information
    log_file_name = "prediction.log"
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

    # Create data frame with SMILES
    new_data = pandas.DataFrame(columns=["target", "smiles"])
    new_data.loc[0] = [numpy.nan, args.new_smiles]

    new_data["smiles_in_training"] = new_data["smiles"].apply(
        smiles_in_training, args=(data,)
    )
    logging.info(f"SMILES in training set: {new_data.loc[0, 'smiles_in_training']} ")

    # Canonical SMILES
    new_data["canonical_smiles"] = new_data["smiles"].apply(smiles_to_canonical)
    # new_data["disconnected_smi"] = new_data["canonical_smiles"].apply(
    #     identify_disconnected_structures
    # )
    # data = data.dropna(axis=0, subset=["disconnected_smi"])
    # data = data.drop(["disconnected_smi", "smiles"], axis=1)

    logging.info(f"Shape of data set: {data.shape} ")

    # ================================
    # String encoding & Augmentation
    # ================================
    time_start_augmenting = datetime.now()

    (
        ml_model,
        augmentation_strategy,
        augmentation_number,
        longest_smiles,
    ) = retrieve_optimal_model(args.task)

    if len(args.new_smiles) > longest_smiles:
        # logging.info(f"The SMILES is too long for this model. Progam aborting")
        raise ValueError("The SMILES is too long for this model. Progam aborting")

    new_data["augmented_smiles"] = new_data["canonical_smiles"].apply(
        augmentation_strategy, args=(augmentation_number,)
    )
    time_end_augmenting = datetime.now()
    time_augmenting = time_end_augmenting - time_start_augmenting
    logging.info(f"Time for augmentation {time_augmenting}")

    # ================================
    # Input processing
    # ================================

    # Replace double symbols
    new_data["new_smiles"] = new_data["augmented_smiles"].apply(char_replacement)

    # Obtain dictionary for these smiles
    smi_dict = ALL_SMILES_CHARACTERS
    logging.info(f"Number of unique characters: {len(smi_dict)} ")
    logging.info(f"String dictionary: {smi_dict} ")

    # TODO
    # Obtain longest of all smiles
    max_length_smi = longest_smiles
    # Check if new smiles are longer that the longest smiles
    logging.info(f"Longest smiles in data set: {max_length_smi} ")

    # ==================================
    # Machine learning ML
    # ==================================

    (ml_model_name, ml_model) = model_type(ml_model, device, smi_dict, max_length_smi)
    logging.info(f"Summary of ml model: {ml_model} ")
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

    all_output_pred = []

    for index, row in new_data.iterrows():
        # Obtain prediction for each of the random smiles of a given molecule
        multiple_output = numpy.concatenate(
            [output_prediction[smiles] for smiles in row["new_smiles"]]
        )

        logging.info(f"Prediction per random SMILES: {multiple_output}")
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
        Draw.MolToFile(molecule, f"maxsmi/prediction/{args.task}/2D_molecule.png")

    new_ensemble_learning.to_csv(f"{folder}/results_out_of_sample.csv")

    time_end_testing = datetime.now()
    time_testing = time_end_testing - time_start_testing
    logging.info(f"Time for model testing {time_testing}")

    time_execution_end = datetime.now()
    time_execution = time_execution_end - time_execution_start
    logging.info(f"Time for model execution {time_execution}")

    logging.info("Script completed. \n \n")
