"""
Unit and regression test for the maxsmi package.
"""

# Import package, test suite, and other packages as needed
import maxsmi
import pytest
import sys

from maxsmi.utils_encoding import get_max_length, char_replacement
from maxsmi.utils_encoding import get_unique_elements_as_dict
from maxsmi.utils_encoding import one_hot_encode

def test_maxsmi_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "maxsmi" in sys.modules

####################
@pytest.mark.parametrize(
    "list_, solution",
    [
        (['1', 'helloworld!','12'], 11),
        (['a', 'b', 'CC1CC1'], 6),
    ],
)

def test_get_max_length(list_, solution):
    max_len = get_max_length(list_)
    assert solution == max_len

####################

@pytest.mark.parametrize(
    "smiles, solution",
    [
        ('Cl', 'L'),
        ('CBrCC', 'CRCC'),
        ('C@@C', 'C$C')
    ],
)

def test_char_replacement(smiles, solution):
    replac_smiles = char_replacement(smiles)
    assert solution == replac_smiles

####################

# TODO

# @pytest.mark.parametrize(
#     "list_, solution",
#     [
#         (['1', '2', 'CCC'], {'1':2, '2':1, 'C':0})
#     ],
# )

#def test_get_unique_elements_as_dict(list_, solution):
#    unique_elem = get_unique_elements_as_dict(list_)
#    assert solution == unique_elem


####################

# TODO
#@pytest.mark.parametrize(
#    "ohe_matrix, solution",
#    [
#
#    ],
#)
#
#def test_one_hot_encode(sequence, dictionary, solution):
#    one_hot = one_hot_encode(sequence, dictionary)
#    assert solution == one_hot
