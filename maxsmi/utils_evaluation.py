"""
utils_evaluation.py
Evaluation metrics.

Handles the primary functions
"""

from sklearn.metrics import r2_score, mean_squared_error


def evaluation_results(output_predicted, output_true):
    """
    Computes metrics on ML predictions.

    Parameters
    ----------
    output_true : array
        Labelled output from the data.
    output_predicted : array
        Predicted output from the model.

    Returns
    -------
    tuple of float
        (mean squared error,
        root mean squared error,
        measure of good-of-fit)
    """
    mean_squared_err = mean_squared_error(output_true, output_predicted, squared=True)
    root_mean_squared_err = mean_squared_error(
        output_true, output_predicted, squared=False
    )
    goodness_of_fit = r2_score(output_true, output_predicted)

    return (mean_squared_err, root_mean_squared_err, goodness_of_fit)
