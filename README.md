Maxsmi: data augmentation for molecular property prediction using deep learning
==============================
[//]: # (Badges)

[![Actions Status](https://github.com/volkamerlab/maxsmi/workflows/CI/badge.svg)](https://github.com/volkamerlab/maxsmi/actions) [![codecov](https://codecov.io/gh/volkamerlab/maxsmi/branch/main/graph/badge.svg)](https://codecov.io/gh/volkamerlab/maxsmi/branch/main) [![Actions Status](https://github.com/volkamerlab/maxsmi/workflows/flake8/badge.svg)](https://github.com/volkamerlab/maxsmi/actions)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/maxsmi/badge/?version=latest)](https://maxsmi.readthedocs.io/en/latest/?badge=latest)

![GitHub closed pr](https://img.shields.io/github/issues-pr-closed-raw/volkamerlab/maxsmi) ![GitHub open pr](https://img.shields.io/github/issues-pr-raw/volkamerlab/maxsmi) ![GitHub closed issues](https://img.shields.io/github/issues-closed-raw/volkamerlab/maxsmi) ![GitHub open issues](https://img.shields.io/github/issues/volkamerlab/maxsmi)

# Table of contents
- Project description
- Installation using conda
  - Prerequisites
  - How to install
- How to use maxsmi
  - Examples
    - How to train and evaluate a model using augmentation
    - How to make predictions
- Documentation
- Repository structure and important files

# Project description
## SMILES augmentation for deep learning based molecular property and activity prediction.

Accurate molecular property or activity prediction is one of the main goals in computer-aided drug design in which deep learning has become an important part. Since neural networks are data greedy and both physico-chemical and bioactivity data sets remain scarce, augmentation techniques have become a powerful assistance for accurate predictions.

This repository provides the code basis to exploit data augmentation using the fact that one compound can be represented by various SMILES (simplified molecular-input line-entry system) strings.

**Augmentation strategies**
* No augmentation
* Augmentation with duplication
* Augmentation without duplication
* Augmentation with reduced duplication
* Augmentation with estimated maximum

**Data sets**
* Physico-chemical data from MoleculeNet, available as part of [DeepChem](https://deepchem.readthedocs.io/en/latest/index.html)
    * ESOL
    * free solv
    * lipophilicity
* Bioactivity data on the EGFR kinase, retrieved from [Kinodata](https://github.com/openkinome/kinodata)

**Deep learning models**
* 1D convolutional neural network (CONV1D)
* 2D convolutional neural network (CONV2D)
* Recurrent neural network (RNN)

The results of our study show that data augmentation improves the accuracy independently of the deep learning model and the size of the data. The best strategy leads to the Maxsmi models, which are available here for predictions on novel compounds on the provided data sets.

# Installation using conda

## Prerequisites
Anaconda and Git should be installed. See [Anaconda's website](https://www.anaconda.com/products/individual) and [Git's website](https://git-scm.com/downloads) for download.

## How to install

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

5. Install the maxsmi package:
```console
pip install -e .
```

# How to use maxsmi
## Examples
### How to train and evaluate a model using augmentation

To get an overview of all available options:

```console
python maxsmi/full_workflow.py --help
```

To train a model with the ESOL data set, augmenting the training set 5 times and the test set 2 times, training for 5 epochs:

```console
python maxsmi/full_workflow.py --task="ESOL" --aug-strategy-train="augmentation_without_duplication" --aug-nb-train=5 --aug-nb-test=2 --nb-epochs 5
```

If no ensemble learning is wanted for the evaluation, add the flag as below:
```console
python maxsmi/full_workflow.py --task="ESOL" --aug-strategy-train="augmentation_without_duplication" --aug-nb-train=5 --aug-nb-test=2 --nb-epochs 5 --eval-strategy=False
```

To train a model with all chosen arguments:

⚠️ This command uses the default number of epochs (which is set to 250). Please allow time for the model to train.

```console
python maxsmi/full_workflow.py --task="free_solv" --string-encoding="smiles" --aug-strategy-train="augmentation_with_duplication" --aug-strategy-test="augmentation_with_reduced_duplication" --aug-nb-train=5 --aug-nb-test=2 --ml-model="CONV1D" --eval-strategy=True --nb-epochs=250
```

### How to make predictions

These predictions use the precalculated `Maxsmi` models (best performing models in the study).

To predict the affinity of a compound against the EGFR kinase, e.g. given by the SMILES `CC1CC1`, run:
```console
python maxsmi/prediction_unlabeled_data.py --task="affinity" --smiles_prediction="CC1CC1"
```

To predict the lipophilicity prediction for the semaxanib drug, run:
```console
python maxsmi/prediction_unlabeled_data.py --task="lipophilicity" --smiles_prediction="O=C2C(\c1ccccc1N2)=C/c3c(cc([nH]3)C)C"
```
# Documentation

The `maxsmi` package documentation is available [here](https://maxsmi.readthedocs.io/en/latest/).


# Repository structure and important files

```
├── LICENSE
├── README.md
├── devtools
├── docs
├── maxsmi
│   ├── augmentation_strategies.py      <- SMILES augmentation strategies
│   ├── full_workflow.py                <- Training and evaluation of deep learning model
│   ├── output_                         <- Saved outputs for results analysis
│   ├── prediction_models               <- Weights for Maxsmi models
│   ├── prediction_unlabeled_data.py    <- Maxsmi models available for user prediction
│   ├── results_analysis                <- Notebooks for results analysis
│   ├── tests

```

### Copyright

Copyright (c) 2020, Talia B. Kimber [@VolkamerLab](https://volkamerlab.org/).


#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.4.
