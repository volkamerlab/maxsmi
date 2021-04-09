"""
parser_default.py
The default parameters for parser.
"""

from maxsmi.augmentation_strategies import augmentation_with_duplication


# Data and related task
TASK = "ESOL"

# Molecular string encoding
STRING_ENCODING = "smiles"

# Augmentation strategy for the train and test sets
AUGMENTATION_STRATEGY = augmentation_with_duplication

# Augmentation on train and test sets
TRAIN_AUGMENTATION = 10
TEST_AUGMENTATION = 10

# The machine learning model
ML_MODEL = "CONV1D"

# If ensemble learning is applied in the evaluation
ENSEMBLE_LEARNING = True
