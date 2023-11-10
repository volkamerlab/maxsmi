Getting Started
===============

**maxsmi: SMILES augmentation for deep learning based molecular property and activity prediction**

Do you want more accurate prediction for molecular properties when training a deep neural network?
Do you need a prediction for a novel compound?
You've come to the right place!

Installation
-------------

Prerequisites
~~~~~~~~~~~~~

Anaconda should be installed. See `Anaconda's website <https://www.anaconda.com>`_ for download.

How to install
~~~~~~~~~~~~~~

1. Create a :code:`conda` environment, called e.g. :code:`maxsmi`

:code:`conda env create -n maxsmi`

2. Activate the environment:

:code:`conda activate maxsmi`

3. Install the :code:`maxsmi` package:

For Linux / MacOS:

:code:`conda install -c conda-forge maxsmi`

For Windows:

:code:`conda install maxsmi -c conda-forge -c defaults`
For installation, please refer to `the Github page <https://github.com/volkamerlab/maxsmi#installation-using-conda>`_.


Citation
--------

If you use ``maxsmi``, don't forget to reference the work. The paper can be found at this `link <https://doi.org/10.1016/j.ailsci.2021.100014>`_.

.. code-block::

    @article{kimber_2021_AILSCI,
        title = {Maxsmi: Maximizing molecular property prediction performance with confidence estimation using SMILES augmentation and deep learning},
        author = {Talia B. Kimber and Maxime Gagnebin and Andrea Volkamer}
        journal = {Artificial Intelligence in the Life Sciences},
        volume = {1},
        pages = {100014},
        year = {2021},
        issn = {2667-3185},
        doi = {https://doi.org/10.1016/j.ailsci.2021.100014},
        url = {https://www.sciencedirect.com/science/article/pii/S2667318521000143}
    }
