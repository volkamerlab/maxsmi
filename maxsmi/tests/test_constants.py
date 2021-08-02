"""
Unit and regression test for the maxsmi package.
"""

# Import package, test suite, and other packages as needed
# import maxsmi
import sys
from maxsmi.constants import (
    TEST_RATIO,
    RANDOM_SEED,
    BACTH_SIZE,
    LEARNING_RATE,
    NB_EPOCHS,
)


def test_maxsmi_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "maxsmi" in sys.modules


def test_constants():
    assert TEST_RATIO == 0.2
    assert RANDOM_SEED == 1234
    assert BACTH_SIZE == 16
    assert LEARNING_RATE == 0.001
    assert NB_EPOCHS == 250
