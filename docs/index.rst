.. Gretel Trainer documentation master file, created by
   sphinx-quickstart on Tue Oct 11 09:08:14 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Gretel Trainer
==============

This module is designed to provide a simple interface to help users successfully train synthetic models on complex datasets with high row and column counts, and offers features such as Cloud SaaS based training and multi-GPU based parallelization. Get started for free with an API key from `Gretel.ai <https://console.gretel.cloud>`_.

Current functionality and features:
-----------------------------------

* Synthetic data generators for text, tabular, and time-series data with the following features:
    * Balance datasets or boost a minority class using Conditional Data Generation.
    * Automated data validation.
    * Synthetic data quality reports.
    * Privacy filters and optional differential privacy support.
* Multiple `model types supported <https://docs.gretel.ai/synthetics/models>`_\:
    * `Gretel-LSTM` model type supports text, tabular, time-series, and conditional data generation.
    * `Gretel-ACTGAN` model type supports tabular and conditional data generation.
    * `Gretel-GPT` natural language synthesis based on an open-source implementation of GPT-3 (coming soon).
    * `Gretel-DGAN` multi-variate time series based on DoppelGANger (coming soon).

Train Synthetic Data in as Little as Three Lines of Code!
---------------------------------------------------------

#. Install the Gretel CLI and Gretel Trainer either on your system or in your Notebook.

   .. code-block:: bash

      # Command line installation
      pip install -U gretel-client gretel-trainer

      # Notebook installation
      !pip install -Uqq gretel-client gretel-trainer

#. Add your `Gretel API <https://console.gretel.cloud>`_ key via the Gretel CLI.

   Use the Gretel client to store your API key to disk. This step is optional, the trainer will prompt for an API key in the next step.

   .. code-block:: bash

      gretel configure

#. Train or fine-tune a model using the Gretel API.

   .. code-block:: python3

      from gretel_trainer import trainer

      dataset = "https://gretel-public-website.s3-us-west-2.amazonaws.com/datasets/USAdultIncome5k.csv"

      model = trainer.Trainer()
      model.train(dataset)

#. Generate synthetic data!

   .. code-block:: python3

      df = model.generate()

Try it out now!
---------------

If you want to quickly get started synthesizing data with **Gretel.ai**, simply click the button below and follow the examples. See additional Python3 and Jupyter Notebook examples in the `./notebooks` folder.

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/gretelai/trainer/blob/main/notebooks/trainer-examples.ipynb
   :alt: Open in Colab

Join the Synthetic Data Community Discord
-----------------------------------------

If you want to be part of the Synthetic Data Community to receive announcements of the latest releases,
ask questions, suggest new features, or participate in the development meetings, please join
the Synthetic Data Community Server!

.. image:: https://img.shields.io/discord/1007817822614847500?label=Discord&logo=Discord
   :target: https://gretel.ai/discord
   :alt: Discord

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Modules
=======

.. toctree::
   :maxdepth: 2

   quickstart.rst
   trainer.rst
   models.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
