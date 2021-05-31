"""
Unit and regression test for the maxsmi package.
"""

# Import package, test suite, and other packages as needed
# import maxsmi
import pytest
import sys
import pandas
import torch
from maxsmi.pytorch_data import AugmentSmilesData, data_to_pytorch_format
from maxsmi.utils_encoding import get_unique_elements_as_dict, get_max_length


def test_maxsmi_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "maxsmi" in sys.modules


####################
pandas_df = pandas.DataFrame({"new_smiles": ["CC", "C", "CO"], "target": [3, 4, 4]})


@pytest.mark.parametrize(
    "pandas_data_frame, solution",
    [
        (pandas_df, 3),
    ],
)
def test___len___(pandas_data_frame, solution):
    smiles_data = AugmentSmilesData(pandas_data_frame, index_augmentation=True)
    result = smiles_data.__len__()
    assert solution == result


@pytest.mark.parametrize(
    "pandas_data_frame, ind, solution",
    [
        (pandas_df, 1, ("C", 4)),
    ],
)
def test___getitem___(pandas_data_frame, ind, solution):
    smiles_data = AugmentSmilesData(pandas_data_frame)
    result = smiles_data.__getitem__(ind)
    assert solution == result


@pytest.mark.parametrize(
    "smiles, target, smiles_dictionary, maximum_length, machine_learning_model_name, device_to_use, per_mol, solution",
    [
        (
            ["CC", "CN"],
            torch.tensor([1.03]),
            get_unique_elements_as_dict(["CC", "CN"]),
            2,
            "CON1D",
            "cpu",
            False,
            torch.tensor([[[1, 1], [0, 0]], [[1, 0], [0, 1]]]).float(),
        ),
        (
            ["CC", "CN"],
            torch.tensor([1.03]),
            get_unique_elements_as_dict(["CC", "CN"]),
            2,
            "CONV2D",
            "cpu",
            False,
            torch.tensor([[[[1, 1], [0, 0]]], [[[1, 0], [0, 1]]]]).float(),
        ),
        (
            ["CC", "CN"],
            torch.tensor([1.03]),
            get_unique_elements_as_dict(["CC", "CN"]),
            2,
            "RNN",
            "cpu",
            False,
            torch.tensor([[[1, 0], [1, 0]], [[1, 0], [0, 1]]]).float(),
        ),
    ],
)
def test_data_to_pytorch_format(
    smiles,
    target,
    smiles_dictionary,
    maximum_length,
    machine_learning_model_name,
    device_to_use,
    per_mol,
    solution,
):
    (input_, output) = data_to_pytorch_format(
        smiles,
        target,
        smiles_dictionary,
        maximum_length,
        machine_learning_model_name,
        device_to_use,
        per_mol,
    )
    assert (solution == input_).all()
