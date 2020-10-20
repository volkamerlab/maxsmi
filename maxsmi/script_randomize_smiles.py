"""
Randomize SMILES

Example with a single SMILES
"""

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--nb_rand', type=int, help='nb_rand will be generated', default=3)
args = parser.parse_args()

from utils_smiles import smi2can, smi2rand, smi2unique_rand

if __name__ == '__main__':

    # ==================== Example with a SMILES ================
    smile = 'Nc2nc1n(COCCO)cnc1c(=O)[nH]2'

    can_smi = smi2can(smile)
    print('Initial SMILES')
    print(can_smi)

    ran_smi = smi2rand(smile, int_aug=args.nb_rand)
    print(f'List of {args.nb_rand} random SMILES')
    print(*ran_smi)

    ran_unique_smi = smi2unique_rand(smile, int_aug=args.nb_rand)
    print(*ran_unique_smi)
    print(f'List of {len(ran_unique_smi)} out of '
        f'{args.nb_rand} generated random SMILES')