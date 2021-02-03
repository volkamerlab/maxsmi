"""
Unit and regression test for the maxsmi package.
"""

# Import package, test suite, and other packages as needed
# import maxsmi
import pytest
import sys
from maxsmi.utils import string_to_bool


def test_maxsmi_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "maxsmi" in sys.modules


####################
@pytest.mark.parametrize(
    "string, solution",
    [("yes", True), ("True", True), ("False", False)],
)
def test_string_to_bool(string, solution):
    result = string_to_bool(string)
    assert solution == result
