.. maxsmi documentation master file, created by
   sphinx-quickstart on Thu Mar 15 13:55:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to maxsmi's documentation!
=========================================================
**SMILES augmentation for deep learning based molecular property and activity prediction**

.. image::
   https://github.com/volkamerlab/maxsmi/workflows/CI/badge.svg
   :target: https://github.com/volkamerlab/maxsmi/actions

.. image::
   https://codecov.io/gh/volkamerlab/maxsmi/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/volkamerlab/maxsmi/branch/main

.. image::
   https://github.com/volkamerlab/maxsmi/workflows/flake8/badge.svg
   :target: https://github.com/volkamerlab/maxsmi/actions

.. image::
   https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT

.. image::
   https://readthedocs.org/projects/maxsmi/badge/?version=latest
   :target: https://maxsmi.readthedocs.io/en/latest/?badge=latest

.. raw:: html

   <p align="center">
   <img src="_static/ensemble_learning_prediction.png" alt="Ensemble learning" width="600"/>
   <br>
   <font size="3">
   Given a compound represented by its canonical SMILES, the Maxsmi model produces a prediction for each of the SMILES variations.
   The aggregation of these values leads to a per compound prediction and the standard deviation to an uncertainty in the prediction.
   </font>
   </p>

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   example/user_prediction
   example/smiles_augmentation




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
