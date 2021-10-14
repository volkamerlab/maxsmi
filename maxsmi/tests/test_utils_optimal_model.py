"""
Unit and regression test for the maxsmi package.
"""

# Import package, test suite, and other packages as needed
# import maxsmi
import pytest
import sys

from maxsmi.utils_optimal_model import retrieve_optimal_model
from maxsmi.augmentation_strategies import (
    augmentation_with_duplication,
    augmentation_without_duplication,
    augmentation_with_reduced_duplication,
)


def test_maxsmi_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "maxsmi" in sys.modules


####################
@pytest.mark.parametrize(
    "task, solution",
    [
        ("FreeSolv", ("CONV1D", augmentation_without_duplication, 70)),
        ("ESOL", ("CONV1D", augmentation_with_reduced_duplication, 70)),
        ("lipo", ("CONV1D", augmentation_with_duplication, 80)),
        ("lipophilicity", ("CONV1D", augmentation_with_duplication, 80)),
        ("chembl28", ("CONV1D", augmentation_with_reduced_duplication, 70)),
        ("affinity", ("CONV1D", augmentation_with_reduced_duplication, 70)),
        ("ewj", ("CONV1D", augmentation_with_reduced_duplication, 70)),
    ],
)
def test_retrieve_optimal_model(task, solution):
    result = retrieve_optimal_model(task)
    assert solution == result
