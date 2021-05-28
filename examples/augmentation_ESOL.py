"""
workflow.py

Given the ESOL data set, generate augmentation.

"""

import argparse
from maxsmi.utils_data import data_retrieval
from maxsmi.utils_smiles import smiles_to_canonical, smiles_to_random

parser = argparse.ArgumentParser()
parser.add_argument("--nb_rand", type=int, help="nb_rand will be generated", default=10)
args = parser.parse_args()

if __name__ == "__main__":

    # Read data
    df = data_retrieval("ESOL")

    # Canonical SMILES
    df["canonical_smiles"] = df["smiles"].apply(smiles_to_canonical)

    # Random SMILES
    df["random_smiles"] = df["canonical_smiles"].apply(smiles_to_random, args=(2,))

    print(df.shape, df.columns)
    print(df.head(2))
