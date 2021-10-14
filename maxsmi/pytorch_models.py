"""
pytorch_models.py
Neural networks for machine learning.

Handles the primary functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Convolutional1DNetwork(nn.Module):
    """
    Builds a 1D convolutional neural network and a feed-forward pass.

    Parameters
    ----------
    nb_char : int, default=53
        Expected number of possible characters
        For SMILES characters, we assume 53.
    max_length : int, default=256
        Maximum length of SMILES, set to 256.
    embedding_shape : int, default=200
        Dimension of the embedding after convolution.
    kernel_shape : int, default=10
        Size of the kernel for the convolution.
    hidden_shape : int, default=100
        Number of units in the hidden layer.
    output_shape : int, default=1
        Size of the last unit.
    activation : torch function, default: relu
        The activation function used in the hidden (only!) layer of the network.
    """

    def __init__(
        self,
        nb_char=53,
        max_length=256,
        embedding_shape=300,
        kernel_shape=10,
        hidden_shape=100,
        output_shape=1,
        activation=F.relu,
    ):
        super(Convolutional1DNetwork, self).__init__()

        self.nb_char = nb_char
        self.max_length = max_length
        self.embedding_shape = embedding_shape
        self.kernel_shape = kernel_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self._activation = activation

        self.convolution = nn.Conv1d(
            in_channels=self.nb_char,
            out_channels=self.embedding_shape,
            kernel_size=self.kernel_shape,
        )
        self.temp = (self.max_length - self.kernel_shape + 1) * self.embedding_shape
        self.fully_connected_1 = nn.Linear(self.temp, self.hidden_shape)
        self.fully_connected_out = nn.Linear(self.hidden_shape, self.output_shape)

    def forward(self, x):
        """
        Defines the forward pass for a given input 'x'
        """
        x = self._activation(self.convolution(x))
        x = torch.flatten(x, 1)
        x = self._activation(self.fully_connected_1(x))
        return self.fully_connected_out(x)


class Convolutional2DNetwork(nn.Module):
    """
    Builds a 2D convolutional neural network and a feed-forward pass.

    Parameters
    ----------
    nb_char : int, default=53
        Expected number of possible characters. For SMILES characters, we assume 53.
    max_length : int, default=256
        Maximum length of SMILES, set to 256.
    in_channels : int, default=1
        Number of channel of the input.
    out_channels : int, default=3
        Number of output channels.
    kernel_shape : int, default=10
        Size of the kernel for the convolution.
    hidden_shape : int, default=100
        Number of units in the hidden layer.
    output_shape : int, default=1
        Size of the last unit.
    activation : torch function, default: relu
        The activation function used in the hidden (only!) layer of the network.
    """

    def __init__(
        self,
        nb_char=53,
        max_length=256,
        in_channels=1,
        out_channels=3,
        kernel_shape=10,
        hidden_shape=100,
        output_shape=1,
        activation=F.relu,
    ):
        super(Convolutional2DNetwork, self).__init__()

        self.nb_char = nb_char
        self.max_length = max_length
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_shape = kernel_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self._activation = activation

        self.convolution = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_shape,
        )

        self.temp1 = self.nb_char - self.kernel_shape + 1
        self.temp2 = self.max_length - self.kernel_shape + 1
        self.temp3 = self.temp1 * self.temp2 * self.out_channels
        self.fully_connected_1 = nn.Linear(self.temp3, self.hidden_shape)
        self.fully_connected_out = nn.Linear(self.hidden_shape, self.output_shape)

    def forward(self, x):
        """
        Defines the forward pass for a given input 'x'
        """
        x = self._activation(self.convolution(x))
        x = torch.flatten(x, 1)
        x = self._activation(self.fully_connected_1(x))
        return self.fully_connected_out(x)


class RecurrentNetwork(nn.Module):
    """
    Builds a recurrent neural network with an LTSM cell and a feed-forward pass.

    Parameters
    ----------
    nb_char : int, default=53
        Expected number of possible characters
        For SMILES characters, we assume 53.
    max_length : int, default=256
        Maximum length of SMILES, set to 256.
    hidden_shape : int, default=128
        Shape of the hidden state.
    output_shape : int, default=1
        Size of the last unit.
    activation : torch function, default: relu
        The activation function used in the hidden (only!) layer of the network.
    """

    def __init__(
        self,
        nb_char=53,
        max_length=256,
        hidden_shape=128,
        output_shape=1,
        activation=F.relu,
    ):
        super(RecurrentNetwork, self).__init__()
        self.nb_char = nb_char
        self.max_length = max_length
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self._activation = activation

        self.lstm = nn.LSTM(
            input_size=self.nb_char,
            hidden_size=self.hidden_shape,
            num_layers=1,
            batch_first=True,
        )

        self.fully_connected_1 = nn.Linear(
            self.hidden_shape * self.max_length, self.hidden_shape
        )
        self.fully_connected_2 = nn.Linear(self.hidden_shape, 64)
        self.fully_connected_out = nn.Linear(64, self.output_shape)

    def forward(self, x):
        """
        Defines the foward pass for a given input 'x'
        """
        lstm_output, _ = self.lstm(x)
        x = torch.reshape(
            lstm_output,
            (lstm_output.shape[0], lstm_output.shape[1] * lstm_output.shape[2]),
        )
        x = self.fully_connected_1(x)
        x = self._activation(self.fully_connected_2(x))
        return self.fully_connected_out(x)


def model_type(model_name, device_to_use, smiles_dictionary, max_length_smiles):
    """
    Instanciates a pytorch machine learning model.

    Parameters
    ----------
    model_name : str
        Name of the machine learning model. It can be either "CON1D", "CONV2D", or "RNN".
    device_to_use : torch.device
        The device to use for model instance, "cpu" or "cuda".
    smiles_dictionary : dict
        The dictionary of SMILES characters.
    max_length_smiles : int
        The length of the longest SMILES.

    Returns
    -------
    tuple of (str, nn.Module):
        The name of the model itself, the instance of the model.
    """

    # Initialize machine learning model
    if model_name == "CONV1D":
        model_instanciated = Convolutional1DNetwork(
            nb_char=len(smiles_dictionary), max_length=max_length_smiles
        )
    elif model_name == "CONV2D":
        model_instanciated = Convolutional2DNetwork(
            nb_char=len(smiles_dictionary), max_length=max_length_smiles
        )
    elif model_name == "RNN":
        model_instanciated = RecurrentNetwork(
            nb_char=len(smiles_dictionary), max_length=max_length_smiles
        )
    else:
        raise Exception("Unknown machine learning model. ")

    model_instanciated.to(device_to_use)

    return (model_name, model_instanciated)


class FeedForwardNetwork(nn.Module):
    """
    Builds a feed-forward (or dense) neural network and a feed-forward pass.

    Parameters
    ----------
    input_shape : int, default=1024
        Length of the input Morgan fingerprint.
    hidden_shape : int, default=100
        Number of units in the hidden layer.
    output_shape : int, default=1
        Size of the last unit.
    activation : torch function, default: relu
        The activation function used in the hidden (only!) layer of the network.
    """

    def __init__(
        self,
        input_shape=1024,
        hidden_shape=100,
        output_shape=1,
        activation=F.relu,
    ):
        super(FeedForwardNetwork, self).__init__()

        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self._activation = activation

        self.fully_connected_1 = nn.Linear(self.input_shape, self.hidden_shape)
        self.fully_connected_2 = nn.Linear(self.hidden_shape, 64)
        self.fully_connected_out = nn.Linear(64, self.output_shape)
        # self.double()

    def forward(self, x):
        """
        Defines the forward pass for a given input 'x'
        """
        x = x.float()
        x = self._activation(self.fully_connected_1(x))
        x = self._activation(self.fully_connected_2(x))
        return self.fully_connected_out(x)
