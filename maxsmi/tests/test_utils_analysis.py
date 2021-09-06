"""
Unit and regression test for the maxsmi package.
"""

# Import package, test suite, and other packages as needed
# import maxsmi
import pytest
import sys

from maxsmi.utils_analysis import load_results


def test_maxsmi_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "maxsmi" in sys.modules


####################
def test_load_results_exception():
    with pytest.raises(FileNotFoundError):
        assert load_results(
            "path",
            "free_solv",
            "augmentation_without_duplication",
            70,
            "aumgentation_without_duplication",
            70,
            "CONV1D",
        )
