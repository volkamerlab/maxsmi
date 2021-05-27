"""
pytorch_training.py

Pytorch training loop and necessary parameters.
"""
import logging
import torch

from maxsmi.pytorch_data import data_to_pytorch_format

BACTH_SIZE = 16
LEARNING_RATE = 0.001
NB_EPOCHS = 5


def model_training(
    data_loader,
    ml_model,
    loss_function,
    optimizer,
    nb_epochs,
    is_cuda,
    len_full_data,
    smi_dict,
    max_length_smi,
    device,
):
    """
    # TODO
    """
    logging.info("========")
    logging.info(f"Training for {nb_epochs} epochs")
    logging.info("========")

    ml_model.train()
    loss_per_epoch = []
    for epoch in range(nb_epochs):
        running_loss = 0.0
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

        loss_per_epoch.append(running_loss / len_full_data)
        if epoch % 10 == 0:
            logging.info(f"Epoch : {epoch + 1} ")

        if is_cuda:
            torch.cuda.empty_cache()

        return loss_per_epoch
