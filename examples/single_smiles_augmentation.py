"""
Randomize SMILES

Example with a single SMILES
"""

import argparse
from maxsmi.utils_smiles import (
    smiles_to_canonical,
    smiles_to_random,
    control_smiles_duplication,
)


parser = argparse.ArgumentParser()
parser.add_argument("--nb_rand", type=int, help="nb_rand will be generated", default=5)
args = parser.parse_args()

if __name__ == "__main__":

    # ==================== Example with a SMILES ================
    smile = "CCC1CC1"

    canonical_smile = smiles_to_canonical(smile)
    print("Initial SMILES:")
    print(canonical_smile)
    print("==============")

    random_smiles = smiles_to_random(smile, int_aug=args.nb_rand)
    print(f"List of {args.nb_rand} random SMILES: \n")
    for random_smile in random_smiles:
        print(f"{random_smile} \n")
    print("==============")

    random_unique_smiles = control_smiles_duplication(random_smiles, lambda x: 1)
    print(
        f"List of {len(random_unique_smiles)} (unique) out of "
        f"{args.nb_rand} generated random SMILES:"
    )
    for unique_smile in random_unique_smiles:
        print(f"{unique_smile} \n")
