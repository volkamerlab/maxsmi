"""
utils_evaluation.py
Evaluation metrics.

Handles the primary functions
"""

import pandas as pd
import numpy
from sklearn.metrics import r2_score
import torch
import torch.nn as nn


def evaluation_results(output_true, output_predicted):
    """
    # TODO
    """
    loss_function = nn.MSELoss()
    loss_value = loss_function(output_predicted, output_true)
    root_loss_value = torch.sqrt(loss_value)
    r2 = r2_score(output_true.detach().numpy(), output_predicted.detach().numpy())

    return (loss_value.item(), root_loss_value.item(), r2)
