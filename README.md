maxsmi: a guide to SMILES augmentation
==============================
[//]: # (Badges)

[![Actions Status](https://github.com/volkamerlab/maxsmi/workflows/CI/badge.svg)](https://github.com/volkamerlab/maxsmi/actions) [![codecov](https://codecov.io/gh/volkamerlab/maxsmi/branch/main/graph/badge.svg)](https://codecov.io/gh/volkamerlab/maxsmi/branch/main) [![Actions Status](https://github.com/volkamerlab/maxsmi/workflows/flake8/badge.svg)](https://github.com/volkamerlab/maxsmi/actions)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/maxsmi/badge/?version=latest)](https://maxsmi.readthedocs.io/en/latest/?badge=latest)

![GitHub closed pr](https://img.shields.io/github/issues-pr-closed-raw/volkamerlab/maxsmi) ![GitHub open pr](https://img.shields.io/github/issues-pr-raw/volkamerlab/maxsmi) ![GitHub closed issues](https://img.shields.io/github/issues-closed-raw/volkamerlab/maxsmi) ![GitHub open issues](https://img.shields.io/github/issues/volkamerlab/maxsmi)


## SMILES augmentation for deep learning based molecular property and activity prediction.

Accurate molecular property or activity prediction is one of the main goals in computer-aided drug design. Especially more recently machine learning/deep learning (DL) have become an important part of this process. Since these algorithms are data greedy, but physico-chemical and bioactivity data sets sizes are still scacre, augmentation techniques are increasingly applied in this context. 

This repository provides the code basis to exploit data augmentation techniques using the fact that one compound can be represented by various SMILES (simplified molecular-input line-entry system) strings.

**Augmentation strategies**
* No augmentation
* Augmentation with duplication
* Augmentation without duplication
* Augmentation with reduced duplication
* Augmentation with estimated maximum

**Data sets used**
* Physico-chemical data from MoleculeNet, available as part of [DeepChem](https://deepchem.readthedocs.io/en/latest/index.html)
    * ESOL
    * free solv
    * lipohilicity
* Bioactivity data from ChEMBL exemplifid on the EGFR kinase, retrieved from [kinodata](https://github.com/openkinome/kinodata)

**DL models**
* 1D convolutional neural network (CONV1D)
* 2D convolutional neural network (CONV2D) 
* Recurrent neural network (RNN)

The results of our study show that data augmentation improves the accuracy independently of the deep learning model and the size of the data. The best strategy led to the maxsmi models, which are available here for predictions on novel compounds on the provided data sets.

## Installation using conda

### Prerequisite
Anaconda and Git should be installed. See [Anaconda's website](https://www.anaconda.com/products/individual) and [Git's website](https://git-scm.com/downloads) for download.

### How to

1. Clone the github repository:
```console
git clone https://github.com/volkamerlab/maxsmi.git
```

2. Change directory:
```console
cd maxsmi
```
3. Create the conda environment:

```console
conda env create -n maxsmi -f devtools/conda-envs/test_env.yaml
```

4. Activate the environment:

```console
conda activate maxsmi
```

5. Install maxsmi package:
```console
pip install -e .
```

## How to use maxsmi

### Documentation

The `maxsmi` package documentation is available [here](https://maxsmi.readthedocs.io/en/latest/).

### Examples how to train a model

To get an overview of all available options:

```console
python maxsmi/full_workflow.py --help
```

To train a model with the ESOL data set, augmenting the training set 5 times and the test set 2 times, training for 5 epochs:

```console
python maxsmi/full_workflow.py --task=ESOL --aug-strategy-train=augmentation_without_duplication --aug-nb-train=5 --aug-nb-test=2 --nb-epochs 5
```

If no ensemble learning is wanted for the evaluation, run:
```console
python maxsmi/full_workflow.py --task=ESOL --eval-strategy=False
```

To train a model with all chosen arguments:
```console
python maxsmi/full_workflow.py --task=free_solv --string-encoding=smiles --aug-strategy-train=augmentation_with_duplication --aug-strategy-test=augmentation_with_reduced_duplication --aug-nb-train=5 --aug-nb-test=2 --ml-model=CONV1D --eval-strategy=True --nb-epochs=250
```

### Examples how to make predictions

Note these predictions use the precalculated `maxsmi` models (best performing models in the study).

To predict the affinity of a compound against the EGFR kinase, e.g. given by the SMILES `CC1CC1`, run:
```console
python maxsmi/prediction_unlabeled_data.py --task="affinity" --smiles_prediction="CC1CC1"
```

To predict the lipophilicity prediction for the semaxanib drug, run:
```console
python maxsmi/prediction_unlabeled_data.py --task="lipophilicity" --smiles_prediction="O=C2C(\c1ccccc1N2)=C/c3c(cc([nH]3)C)C"
```

### Copyright

Copyright (c) 2020, Talia B. Kimber @ Volkamer Lab


#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.4.
