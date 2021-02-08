"""
Unit and regression test for the maxsmi package.
"""

# Import package, test suite, and other packages as needed
# import maxsmi
import pytest
import sys
import pandas
from maxsmi.pytorch_data import AugmenteSmilesData


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
    smiles_data = AugmenteSmilesData(pandas_data_frame, index_augmentation=True)
    result = smiles_data.__len__()
    assert solution == result


@pytest.mark.parametrize(
    "pandas_data_frame, ind, solution",
    [
        (pandas_df, 1, ("C", 4)),
    ],
)
def test___getitem___(pandas_data_frame, ind, solution):
    smiles_data = AugmenteSmilesData(pandas_data_frame)
    result = smiles_data.__getitem__(ind)
    assert solution == result
