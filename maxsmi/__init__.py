"""
maxsmi
Find the optimal SMILES augmentation for accurate prediction.
"""

# Add imports here
# from . import utils_data, utils_encoding, utils_evaluation, utils_smiles


# Handle versioneer
from ._version import get_versions

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
