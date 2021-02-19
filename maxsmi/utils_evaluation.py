"""
utils_evaluation.py
Evaluation metrics.

Handles the primary functions
"""

from sklearn.metrics import r2_score
import torch
import torch.nn as nn


def evaluation_results(output_true, output_predicted, cuda_available):
    """
    Computes metrics on ML predictions.

    Parameters
    ----------
    output_true : torch.tensor
        Labelled output from the data.
    output_predicted : torch.tensor
        Predicted output from the model.
    cuda_available : bool
        Is CUDA available.

    Returns
    -------
    tuple
        (mean squared error,
        root mean squared error,
        measure of good-of-fit)
    """
    loss_function = nn.MSELoss()
    loss_value = loss_function(output_predicted, output_true)
    root_loss_value = torch.sqrt(loss_value)
    if cuda_available:
        r2 = r2_score(
            output_true.cpu().detach().numpy(), output_predicted.cpu().detach().numpy()
        )
    else:
        r2 = r2_score(output_true.detach().numpy(), output_predicted.detach().numpy())

    return (loss_value.item(), root_loss_value.item(), r2)
