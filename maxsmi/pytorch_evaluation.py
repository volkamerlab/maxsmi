"""
pytorch_evaluation.py

Pytorch evalution.
"""
import torch

from maxsmi.pytorch_data import data_to_pytorch_format


def model_evaluation(
    data_loader,
    ml_model,
    smi_dict,
    max_length_smi,
    device,
):
    """
    # TODO
    """
    ml_model.eval()
    with torch.no_grad():
        output_pred_train = []
        output_true_train = []
        for i, data in enumerate(data_loader):
            # SMILES and target
            smiles, target = data

            input_true, output_true = data_to_pytorch_format(
                smiles,
                target,
                smi_dict,
                max_length_smi,
                ml_model,
                device,
            )

            # Prediction
            output_pred = ml_model(input_true)
            output_pred_train.append(output_pred)
            output_true_train.append(output_true)

        output_pred_train = torch.stack(output_pred_train)
        output_true_train = torch.stack(output_true_train)
        output_pred_train = output_pred_train.view(-1, 1)
        output_true_train = output_true_train.view(-1, 1)
    return (output_true_train, output_pred_train)
