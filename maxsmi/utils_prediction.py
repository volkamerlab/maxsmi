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

    elif task == "lipho" or "lipophilicity":
        ml_model = "CONV1D"
        augmentation_strategy = augmentation_with_duplication
        augmentation_number = 5
        longest_smiles = 268

    elif task == "chembl28" or "affinity":
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
