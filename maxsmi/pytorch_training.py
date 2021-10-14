"""
pytorch_training.py

Pytorch training loop and necessary parameters.
"""
import logging
import torch
import torch.optim as optim

from maxsmi.pytorch_data import data_to_pytorch_format


def model_training(
    data_loader_train,
    data_loader_valid,
    ml_model_name,
    ml_model,
    loss_function,
    nb_epochs,
    is_cuda,
    smiles_dictionary,
    max_length_smiles,
    device_to_use,
    learning_rate,
    path_to_parameters,
    patience=3,
):
    """
    # Train the machine learning model using the optimization loop in pytorch.

    Parameters
    ----------
    data_loader_train : torch.utils.data
        The training data as seen by Pytorch for mini-batches.
    data_loader_valid : torch.utils.data
        The validation data as seen by Pytorch for mini-batches.
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
    smiles_dictionary : dict
        The dictionary of SMILES characters.
    max_length_smiles : int
        The length of the longest SMILES.
    device_to_use : torch.device
        The device to use for model instance, "cpu" or "cuda".
    learning_rate : float
        The learning rate in the optimization algorithm.
    path_to_parameters : str
        The path to save the model parameters.
    patience : int, default=10
        Number of epochs to wait since the last time validation loss improved.
    Returns
    -------
    # TODO
    tuple
        The loss value at each epoch.
    """

    optimizer = optim.SGD(ml_model.parameters(), lr=learning_rate)

    epoch = 0
    epoch_loss_train = 0.0
    epoch_loss_valid = 0.0

    loss_train_list = []
    loss_valid_list = []

    # Number of data points in the training/validation set
    len_train_data = len(data_loader_train.dataset)
    len_valid_data = len(data_loader_valid.dataset)

    while epoch <= nb_epochs:
        ################
        # Train the model
        ################
        ml_model.train()
        batch_loss_train = 0.0
        for _, data in enumerate(data_loader_train):

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
            # Save batch loss
            batch_loss_train += float(loss.item())
            # free memory
            del data

        ################
        # Validate the model
        ################
        ml_model.eval()
        batch_loss_valid = 0.0
        for _, data in enumerate(data_loader_valid):

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

            output_pred = ml_model(input_true)
            loss = loss_function(output_pred, output_true)
            batch_loss_valid += float(loss.item())

        epoch_loss_train = batch_loss_train / len_train_data
        epoch_loss_valid = batch_loss_valid / len_valid_data

        # if epoch % 10 == 0:
        #    logging.info(f"Epoch : {epoch + 1} ")
        # TODO: add logging for early stopping
        # print(f"Epoch : {epoch} ")
        # print(f"Train loss: {epoch_loss_train:.3f}")
        # print(f"Valid loss: {epoch_loss_valid:.3f}")
        logging.info(f"Epoch: {epoch} ")
        logging.info(f"Train loss: {epoch_loss_train:.3f}")
        logging.info(f"Valid loss: {epoch_loss_valid:.3f}")

        loss_train_list.append(epoch_loss_train)
        loss_valid_list.append(epoch_loss_valid)

        # Clear memory
        if is_cuda:
            torch.cuda.empty_cache()

        # Update epoch
        epoch += 1

        #######
        # Early stopping
        #######

        # Initialize early stopping:
        if epoch == 1:
            best_loss_valid = epoch_loss_valid
            waiting = 0
        # Check for early stopping criteria
        if epoch_loss_valid > best_loss_valid:
            waiting += 1
        else:
            best_loss_valid = epoch_loss_valid
            waiting = 0
            # Save best model
            torch.save(ml_model.state_dict(), f"{path_to_parameters}/model_dict.pth")
        if waiting > patience:
            # print(f"Early stopping at epoch {epoch}.")
            logging.info(f"Early stopping at epoch: {epoch} ")
            # ml_model = torch.load(f"{path_to_parameters}/model_dict.pth")
            torch.save(ml_model.state_dict(), f"{path_to_parameters}/model_dict.pth")
            break

    ml_model.load_state_dict(torch.load(f"{path_to_parameters}/model_dict.pth"))
    # print(f"Effective number of epochs for training {epoch}.")
    # print("Train error per epoch:", loss_train_list)
    # print("Valid error per epoch:", loss_valid_list)
    logging.info(f"Effective number of epochs for training {epoch}.")
    # logging.info("Train error per epoch:", loss_train_list)
    # logging.info("Valid error per epoch:", loss_valid_list)
    return ml_model, loss_train_list, loss_valid_list
