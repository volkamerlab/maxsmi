"""
prediction_optimal.py
Retrieves optimal prediction for each task.
"""
import logging
from maxsmi.augmentation_strategies import augmentation_with_duplication


def retrieve_optimal_model(task):
    """
    #TODO.

    Parameters
    ----------
    task : str

    Returns
    -------
    """
    if task == "free_solv":
        ml_model = "CONV1D"
        augmentation_strategy = augmentation_with_duplication
        augmentation_number = 5
        longest_smiles = 74

    elif task == "ESOL":
        ml_model = "CONV1D"
        augmentation_strategy = augmentation_with_duplication
        augmentation_number = 5
        longest_smiles = 110

    elif task in ["lipho", "lipophilicity"]:
        ml_model = "CONV1D"
        augmentation_strategy = augmentation_with_duplication
        augmentation_number = 5
        longest_smiles = 268

    elif task in ["chembl28", "affinity"]:
        ml_model = "CONV1D"
        augmentation_strategy = augmentation_with_duplication
        augmentation_number = 5
        longest_smiles = 74
    else:
        if task != "ESOL":
            logging.warning("Invalid data. Choosing ESOL by default.")
            print(
                "Task unknown. Please choose between free_solv, ESOL, lipophilicity or affinity"
            )

    return (ml_model, augmentation_strategy, augmentation_number, longest_smiles)


def unlabeled_smiles_max_length(unlabeled_smiles, maximum_length_smiles):
    """
    Checks whether the unlabeled SMILES for prediction is longer than the longest SMILES in the training dataset.

    Parameters
    ----------
    unlabeled_smiles : str
        SMILES string describing a compound.

    maximum_length_smiles : int
        The longest SMILES in the dataset on which the model was trained on.
    """
    if len(unlabeled_smiles) > maximum_length_smiles:
        raise ValueError("The SMILES is too long for this model. Program aborting")
    else:
        pass


def mixture_check(unlabeled_smiles):
    """
    Aborts the prediction if the SMILES contains mixtures.

    Parameters
    ----------
    unlabeled_smiles : str
        SMILES string describing a compound.

    Returns
    -------
    str :
        the SMILES if it's not disconnected. Raises an error otherwise.

    """
    if "." in unlabeled_smiles:
        raise ValueError(
            "SMILES containing mixtures cannot be processed. Program aborting"
        )
    else:
        return unlabeled_smiles
