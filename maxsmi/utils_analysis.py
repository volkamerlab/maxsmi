"""
utils_analysis.py
Analysis of results.

Handles the functions needed for the analysis of the results from the simulations
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np


def load_data(
    task,
    augmentation_strategy_train,
    train_augmentation,
    augmentation_strategy_test,
    test_augmentation,
    ml_model,
    string_encoding="smiles",
):
    """
    Loads the data from simulations.

    Parameters
    ----------
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
    data: pd.Pandas
        Pandas data frame with performance metrics (on train and test sets), such as r2 score and time.

    """
    with open(
        f"output/{task}_{string_encoding}_{augmentation_strategy_train}_"
        f"{train_augmentation}_{augmentation_strategy_test}_"
        f"{test_augmentation}_{ml_model}/"
        f"results_metrics.pkl",
        "rb",
    ) as f:
        data = pickle.load(f)
    return data


def plot_loss(
    augmentation_strategy_train,
    train_augmentation,
    augmentation_strategy_test,
    test_augmentation,
    string_encoding="smiles",
):
    """
    Plots the loss as a function of the number of epochs.

    Parameters
    ----------
    augmentation_strategy_train : str
        The augmentation strategy used on the train set.
    train_augmentation : int
        The number of augmentation on the train set.
    augmentation_strategy_test : str
        The augmentation strategy used on the test set.
    test_augmentation : int
        The number of augmentation on the test set.
    string_encoding : str
        The molecular encoding, default is "smiles".

    Returns
    -------
    None
    """
    _, ax = plt.subplots()
    legend_ = []
    for task in ["ESOL", "free_solv"]:
        for model in ["CONV1D", "CONV2D", "RNN"]:
            data = load_data(
                task,
                augmentation_strategy_train,
                train_augmentation,
                augmentation_strategy_test,
                test_augmentation,
                model,
                string_encoding,
            )
            plt.plot(data.loss[0])
            caption = f"{task};{model}"
            legend_.append(caption)

    plt.legend(legend_)

    ax.set_xlabel("Number of epochs")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss for {augmentation_strategy_train}")

    plt.legend(legend_)
    plt.show()
    return None


def retrieve_metric(
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
    data = load_data(
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


def plot_metric(metric, set_, augmentation_strategy, max_augmentation=100):
    """
    Plots the metric of interest on the set of interest.

    Parameters
    ----------
    metric : str
        The metric of interest, such as the r2 score, time or mean squared error.
    set_ : str
        The train set or test set.
    augmentation_strategy : str
            The augmentation strategy used.
    max_augmentation : int, default is 100.
        The largest number of augmentation that was performed.

    Returns
    -------
    None
    """
    _, ax = plt.subplots()

    x = [elem for elem in range(0, max_augmentation + 10, 10)]
    legend_ = []

    for task in ["ESOL", "free_solv"]:
        for model in ["CONV1D", "CONV2D", "RNN"]:
            y_task_model = []
            for augmentation_num in range(0, max_augmentation + 10, 10):
                y = retrieve_metric(
                    metric,
                    set_,
                    task,
                    augmentation_strategy,
                    augmentation_num,
                    augmentation_strategy,
                    augmentation_num,
                    model,
                )
                y_task_model.append(y)
            plt.plot(x, y_task_model)
            plt.legend([task, model])

            caption = f"{task};{model}"
            if metric == "r2":
                print(
                    f"{caption} \t"
                    f"max value: {np.max(np.array(y_task_model)):.2f},"
                    f"\t {np.argmax(np.array(y_task_model))*10}"
                )
            else:
                print(
                    f"{caption}, \t"
                    f"max value: {np.min(np.array(y_task_model)):.2f},"
                    f"\t {np.argmin(np.array(y_task_model))*10}"
                )
            legend_.append(caption)

    plt.legend(legend_)

    ax.set_xlabel("Number of augmentation")
    ax.set_ylabel(f"{metric}")
    ax.set_title(f"{set_} metric {augmentation_strategy}")

    plt.show()

    return None
