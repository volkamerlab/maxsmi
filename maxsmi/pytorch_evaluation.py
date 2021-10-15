"""
pytorch_evaluation.py

Pytorch evalution.
"""
import torch

from maxsmi.pytorch_data import data_to_pytorch_format


def model_evaluation(
    data_loader,
    ml_model_name,
    ml_model,
    smiles_dictionary,
    max_length_smiles,
    device_to_use,
):
    """
    Evaluation per batch of a pytorch machine learning model.

    Parameters
    ----------
    data_loader : torch.utils.data
        The training data as seen by Pytorch for mini-batches.
    ml_model_name : str
        Name of the machine learning model. It can be either "CON1D", "CONV2D", or "RNN".
    ml_model : nn.Module
        Instance of the pytorch machine learning model.
    smiles_dictionary : dict
        The dictionary of SMILES characters.
    max_length_smiles : int
        The length of the longest SMILES.
    device_to_use : torch.device
        The device to use for model instance, "cpu" or "cuda".

    Returns
    -------
    tuple of dict:
        Dictionary of the predicted, true output values, respectively, in the data loader, with SMILES as keys.
    """

    ml_model.eval()
    with torch.no_grad():
        all_output_pred = {}
        all_output_true = {}
        for _, data in enumerate(data_loader):
            # SMILES and target
            smiles, target = data

            input_true, output_true = data_to_pytorch_format(
                smiles,
                target,
                smiles_dictionary,
                max_length_smiles,
                ml_model_name,
                device_to_use,
            )

            # Prediction
            output_pred = ml_model(input_true)

            # Convert to numpy arrays
            output_pred = output_pred.cpu().detach().numpy()
            output_true = output_true.cpu().detach().numpy()

            for smile in smiles:
                all_output_pred[smile] = output_pred
                all_output_true[smile] = output_true

    return (all_output_pred, all_output_true)


def out_of_sample_prediction(
    data_loader,
    ml_model_name,
    ml_model,
    smiles_dictionary,
    max_length_smiles,
    device_to_use,
):
    """
    Prediction using trained pytorch machine learning model.

    Parameters
    ----------
    data_loader : torch.utils.data
        The unlabeled data as seen by Pytorch.
    ml_model_name : str
        Name of the machine learning model. It can be either "CON1D", "CONV2D", or "RNN".
    ml_model : nn.Module
        Instance of the pytorch machine learning model.
    smiles_dictionary : dict
        The dictionary of SMILES characters.
    max_length_smiles : int
        The length of the longest SMILES.
    device_to_use : torch.device
        The device to use for model instance, "cpu" or "cuda".

    Returns
    -------
    np.array:
        The prediction using the trained model.
    """

    ml_model.eval()
    with torch.no_grad():
        all_output = {}
        for _, data in enumerate(data_loader):
            # SMILES and target
            smiles, target = data

            input_true, _ = data_to_pytorch_format(
                smiles,
                target,
                smiles_dictionary,
                max_length_smiles,
                ml_model_name,
                device_to_use,
            )

            # Prediction
            output_pred = ml_model(input_true)

            # Convert to numpy arrays
            output_pred = output_pred.cpu().detach().numpy()
            for smile in smiles:
                all_output[smile] = output_pred

    return all_output


def model_evaluation_fingerprint(data_loader, ml_model):
    """
    Evaluation per batch of a pytorch machine learning model.

    Parameters
    ----------
    data_loader : torch.utils.data
        The training data as seen by Pytorch for mini-batches.
    ml_model : nn.Module
        Instance of the pytorch machine learning model.

    Returns
    -------
    tuple of array:
        Arrays of the predicted, true output values, respectively, in the data loader.
    """

    ml_model.eval()
    with torch.no_grad():
        all_output_pred = []
        all_output_true = []
        for _, data in enumerate(data_loader):
            # fingerprint and target
            fingerprint, output_true = data

            # Prediction
            output_pred = ml_model(fingerprint)

            # Convert to numpy arrays
            output_pred = output_pred.cpu().detach().numpy()
            output_true = output_true.cpu().detach().numpy()
            all_output_pred.append(output_pred)
            all_output_true.append(output_true)

    return (all_output_pred, all_output_true)
