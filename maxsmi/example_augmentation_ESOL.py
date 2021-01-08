"""
workflow.py

Given the ESOL data set, generate augmentation.

"""

import argparse
from utils_data import data_retrieval
from utils_smiles import smi2can, smi2rand, smi2unique_rand

parser = argparse.ArgumentParser()
parser.add_argument("--nb_rand", type=int, help="nb_rand will be generated", default=0)
args = parser.parse_args()

if __name__ == "__main__":

    # Read data
    df = data_retrieval("ESOL")

    # Canonical SMILES
    df["canonical_smiles"] = df["smiles"].apply(smi2can)

    # Random SMILES
    df["random_smiles"] = df["canonical_smiles"].apply(smi2rand, args=(2,))

    # Unique random SMILES
    df["random_unique_smiles"] = df["canonical_smiles"].apply(
        smi2unique_rand, args=(2,)
    )

    print(df.shape, df.columns)
    print(df.head(2))
