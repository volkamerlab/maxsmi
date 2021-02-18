maxsmi: a guide to SMILES augmentation
==============================
[//]: # (Badges)

[![Actions Status](https://github.com/t-kimber/maxsmi/workflows/CI/badge.svg)](https://github.com/t-kimber/maxsmi/actions)

[![codecov](https://codecov.io/gh/t-kimber/maxsmi/branch/main/graph/badge.svg)](https://codecov.io/gh/t-kimber/maxsmi/branch/main)

[![Actions Status](https://github.com/t-kimber/maxsmi/workflows/flake8/badge.svg)](https://github.com/t-kimber/maxsmi/actions)



Find the optimal SMILES augmentation for accurate prediction.

#### Conda installation
Create a conda environment:

```sh
$ conda env create -n maxsmi -f devtools/conda-envs/test_env.yaml
```

Activate the environment:

```sh
$ conda activate maxsmi
```

#### Example

To run an example with the ESOL data set, augmenting the train set 5 times and the test set 2 times:

```console
(maxsmi) $ python maxsmi/full_workflow.py --task ESOL --aug-strategy-train augmentation_without_duplication --aug-nb-train 5
```

If no ensemble learning is wanted for the evaluation, run:
```console
(maxsmi) $ python maxsmi/full_workflow.py --task ESOL --aug-nb-train 5 --aug-nb-test 2 --eval-strategy False
```

To compare SMILES encoding with DeepSMILES, run:
```console
(maxsmi) $ python maxsmi/full_workflow.py --task ESOL --string-encoding deepsmiles
```

Similarly for SELFIES, run:
```console
(maxsmi) $ python maxsmi/full_workflow.py --task ESOL --string-encoding selfies
```

To run an example with all chosen arguments:
```console
(maxsmi) $ python maxsmi/full_workflow.py --task free_solv --string-encoding smiles --aug-strategy-train augmentation_with_duplication --aug-strategy-test augmentation_with_reduced_duplication --aug-nb-train 5 --aug-nb-test 2 --ml-model CONV1D --eval-strategy True
```

### Copyright

Copyright (c) 2020, Talia B. Kimber


#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.4.
