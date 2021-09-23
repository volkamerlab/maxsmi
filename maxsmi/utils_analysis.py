"""
utils_analysis.py
Analysis of results.

Handles the functions needed for the analysis of the results from the simulations
"""

import pickle


def load_results(
    path,
    task,
    augmentation_strategy_train,
    train_augmentation,
    augmentation_strategy_test,
    test_augmentation,
    ml_model,
    string_encoding="smiles",
    ensemble_learning=False,
):
    """
    Loads the result data from simulations.

    Parameters
    ----------
    path : str
        The path to output folder.
    task : str
        The data with associated task, e.g. "ESOL", "free_solv"
    augmentation_strategy_train : str
        The augmentation strategy used on the train set.
    train_augmentation : int
        The number of augmentation on the train set.
    augmentation_strategy_test : str
        The augmentation strategy used on the test set.
    test_augmentation : int
        The number of augmentation on the test set.
    ml_model : str
        The machine learning model, e.g. "CONV1D".
    string_encoding : str
        The molecular encoding, default is "smiles".
    ensemble_learning : bool, default False.
        Whether the results from ensemble learning should be loaded.

    Returns
    -------
    data: pd.Pandas
        Pandas data frame with performance metrics (on train and test sets), such as r2 score and time.

    """
    if ensemble_learning:
        try:
            with open(
                f"{path}/output/{task}_{string_encoding}_{augmentation_strategy_train}_"
                f"{train_augmentation}_{augmentation_strategy_test}_"
                f"{test_augmentation}_{ml_model}/"
                f"results_ensemble_learning.pkl",
                "rb",
            ) as f:
                data = pickle.load(f)
        except FileNotFoundError:
            with open(
                f"{path}/output_/{task}_{string_encoding}_{augmentation_strategy_train}_"
                f"{train_augmentation}_{augmentation_strategy_test}_"
                f"{test_augmentation}_{ml_model}/"
                f"results_ensemble_learning.pkl",
                "rb",
            ) as f:
                data = pickle.load(f)
    else:
        with open(
            f"{path}/output/{task}_{string_encoding}_{augmentation_strategy_train}_"
            f"{train_augmentation}_{augmentation_strategy_test}_"
            f"{test_augmentation}_{ml_model}/"
            f"results_metrics.pkl",
            "rb",
        ) as f:
            data = pickle.load(f)
    return data


def retrieve_metric(
    path,
    metric,
    set_,
    task,
    augmentation_strategy_train,
    train_augmentation,
    augmentation_strategy_test,
    test_augmentation,
    ml_model,
    string_encoding="smiles",
):
    """
    Retrieves a metric of interest on the train or test set.

    Parameters
    ----------
    path : str
        The path to output folder.
    metric : str
        The metric of interest, such as the r2 score, time or mean squared error.
    set_ : str
        The train set or test set.
    task : str
        The data with associated task, e.g. "ESOL", "free_solv"
    augmentation_strategy_train : str
        The augmentation strategy used on the train set.
    train_augmentation : int
        The number of augmentation on the train set.
    augmentation_strategy_test : str
        The augmentation strategy used on the test set.
    test_augmentation : int
        The number of augmentation on the test set.
    ml_model : str
        The machine learning model, e.g. "CONV1D".
    string_encoding : str
        The molecular encoding, default is "smiles".

    Returns
    -------
    float
        The metric of interest on the set of interest.
    """
    data = load_results(
        path,
        task,
        augmentation_strategy_train,
        train_augmentation,
        augmentation_strategy_test,
        test_augmentation,
        ml_model,
        string_encoding,
    )
    if set_ == "test":
        if metric == "r2":
            return data.test[0][2]
        if metric == "mse":
            return data.test[0][0]
        if metric == "rmse":
            return data.test[0][1]
        if metric == "time":
            time = data.time_testing[0]
            return time.seconds

    if set_ == "train":
        if metric == "r2":
            return data.train[0][2]
        if metric == "mse":
            return data.train[0][0]
        if metric == "rmse":
            return data.train[0][1]
        if metric == "time":
            time = data.time_training[0]
            return time.seconds
