"""
pytorch_training.py

Pytorch training loop and necessary parameters.
"""
import logging
import torch
import torch.optim as optim

from maxsmi.pytorch_data import data_to_pytorch_format


def model_training(
    data_loader,
    ml_model_name,
    ml_model,
    loss_function,
    nb_epochs,
    is_cuda,
    len_train_data,
    smiles_dictionary,
    max_length_smiles,
    device_to_use,
    learning_rate,
):
    """
    # Train the machine learning model using the otpimization loop in pytorch.

    Parameters
    ----------
    data_loader : torch.utils.data
        The training data as seen by Pytorch for mini-batches.
    ml_model_name : str
        Name of the machine learning model. It can be either "CON1D", "CONV2D", or "RNN".
    ml_model : nn.Module
        Instance of the pytorch machine learning model.
    loss_function : torch.nn.modules.loss
        The loss function to be optimized.
    nb_epochs : int
        Number of epochs to train the model.
    is_cuda : bool
        Is cuda available.
    len_train_data :
        Number of data points in the training set.
    smiles_dictionary : dict
        The dictionary of SMILES characters.
    max_length_smiles : int
        The length of the longest SMILES.
    device_to_use : torch.device
        The device to use for model instance, "cpu" or "cuda".
    learning_rate : float
        The learning rate in the optimization algorithm.

    Returns
    -------
    list
        The loss value at each epoch.
    """

    optimizer = optim.SGD(ml_model.parameters(), lr=learning_rate)

    ml_model.train()
    loss_per_epoch = []
    for epoch in range(nb_epochs):
        running_loss = 0.0
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

            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward
            output_pred = ml_model(input_true)
            # Objective
            loss = loss_function(output_pred, output_true)
            # Backward
            loss.backward()
            # Optimization
            optimizer.step()
            # Save loss
            running_loss += float(loss.item())
            # free memory
            del data

        loss_per_epoch.append(running_loss / len_train_data)
        if epoch % 10 == 0:
            logging.info(f"Epoch : {epoch + 1} ")

        if is_cuda:
            torch.cuda.empty_cache()

    return loss_per_epoch
