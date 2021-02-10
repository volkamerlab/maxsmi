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
        Defines the foward pass for a given input 'x'
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
        Expected number of possible characters
        For SMILES characters, we assume 53.
    max_length : int, default=256
        Maximum length of SMILES, set to 256.
    output_shape : int, default=1
        Size of the last unit.
    activation : torch function, default: relu
        The activation function used in the hidden (only!) layer of the network.
    """

    def __init__(
        self,
        nb_char=53,
        max_length=256,
        output_shape=1,
        activation=F.relu,
    ):
        super(Convolutional2DNetwork, self).__init__()
        # TODO

    def forward(self, x):
        """
        Defines the foward pass for a given input 'x'
        """
        pass


class RecurrentNetwork(nn.Module):
    """
    Builds a recurrent neural network and a feed-forward pass.

    Parameters
    ----------
    nb_char : int, default=53
        Expected number of possible characters
        For SMILES characters, we assume 53.
    max_length : int, default=256
        Maximum length of SMILES, set to 256.
    output_shape : int, default=1
        Size of the last unit.
    activation : torch function, default: relu
        The activation function used in the hidden (only!) layer of the network.
    """

    def __init__(
        self,
        nb_char=53,
        max_length=256,
        output_shape=1,
        activation=F.relu,
    ):
        super(RecurrentNetwork, self).__init__()
        # TODO

    def forward(self, x):
        """
        Defines the foward pass for a given input 'x'
        """
        pass
