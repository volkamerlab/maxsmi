"""
utils_optimal_model.py

Define the best model for each data set.
"""
import logging
from maxsmi.augmentation_strategies import (
    augmentation_with_duplication,
    augmentation_without_duplication,
    augmentation_with_reduced_duplication,
)


def retrieve_optimal_model(task):
    """
    From the results of the simulations, we retrieve "manually" the best model for each task,
    namely the best augmentation number, augmentation strategy and machine learning model.

    Parameters
    ----------
    task : str
        The task to consider.

    Returns
    -------
    tuple : (str, function, int)
        The ML model, the augmentation strategy, the augmentation number

    """
    if task == "free_solv":
        ml_model = "CONV1D"
        augmentation_strategy = augmentation_without_duplication
        augmentation_number = 70

    elif task == "ESOL":
        ml_model = "CONV1D"
        augmentation_strategy = augmentation_with_reduced_duplication
        augmentation_number = 70

    elif task in ["lipo", "lipophilicity"]:
        ml_model = "CONV1D"
        augmentation_strategy = augmentation_with_duplication
        augmentation_number = 80

    elif task in ["chembl28", "affinity"]:
        ml_model = "CONV1D"
        augmentation_strategy = augmentation_with_reduced_duplication
        augmentation_number = 70
    else:
        if task != "ESOL":
            logging.warning("Invalid data. Choosing ESOL by default.")
            print(
                "Task unknown. Please choose between free_solv, ESOL, lipophilicity or affinity"
            )

    return (ml_model, augmentation_strategy, augmentation_number)
