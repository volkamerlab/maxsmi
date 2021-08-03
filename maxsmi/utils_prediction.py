"""
prediction_optimal.py
Retrieves optimal prediction for each task.
"""
from maxsmi.utils_smiles import ALL_SMILES_CHARACTERS


def retrieve_longest_smiles_from_optimal_model(task):
    """
    From the optimal models that were trained on the full data set using `full_working_optimal.py`,
    we retrieve the longest SMILES that was generated.

    Parameters
    ----------
    task : str
        The task to consider.

    Returns
    -------
    int :
        The longest SMILES that was generated when training the best model strategy for `task` data.

    """
    if task == "free_solv":
        longest_smiles = 76

    elif task == "ESOL":
        longest_smiles = 109

    elif task in ["lipo", "lipophilicity"]:
        longest_smiles = 268

    elif task in ["chembl28", "affinity"]:
        longest_smiles = 246
    else:
        longest_smiles = None

    return longest_smiles


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
        return None


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


def character_check(unlabeled_smiles):
    """
    Aborts the prediction if the SMILES contains an unknown character.

    Parameters
    ----------
    unlabeled_smiles : str
        SMILES string describing a compound.

    Returns
    -------
    str :
        the SMILES if all characters are known, raises an error otherwise.

    """
    for character in unlabeled_smiles:
        if character not in ALL_SMILES_CHARACTERS:
            raise ValueError("SMILES contains unknown character. Program aborting")
    else:
        return unlabeled_smiles
