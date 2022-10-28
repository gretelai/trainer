Quickstart
==========

Initial Setup
-------------

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

Train Synthetic Data
--------------------
.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/gretelai/trainer/blob/main/notebooks/trainer-examples.ipynb
    :alt: Open in Colab

#. Train or fine-tune a model using the Gretel API.

    .. code-block:: python3

        from gretel_trainer import trainer

        dataset = "https://gretel-public-website.s3-us-west-2.amazonaws.com/datasets/USAdultIncome5k.csv"

        model = trainer.Trainer()
        model.train(dataset)

#. Generate synthetic data!

    .. code-block:: python3

        df = model.generate()

Conditional Data Generation
---------------------------
.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/gretelai/trainer/blob/main/notebooks/simple-conditional-generation.ipynb
    :alt: Open in Colab

#. Load and preview the dataset, and set seed fields.

    .. code-block:: python3

        # Load and preview the patient dataset
        import pandas as pd
        from gretel_trainer import trainer

        DATASET_PATH = 'https://gretel-public-website.s3.amazonaws.com/datasets/mitre-synthea-health.csv'
        SEED_FIELDS = ["RACE", "ETHNICITY", "GENDER"]

        print("\nPreviewing real world dataset\n")
        pd.read_csv(DATASET_PATH)

#. Train the model.

    .. code-block:: python3

        # Train model
        model = trainer.Trainer()
        model.train(DATASET_PATH, seed_fields=SEED_FIELDS)

#. Conditionally generate data.

    .. code-block:: python3

        # Conditionally generate data
        seed_df = pd.DataFrame(data=[
            ["black", "african", "F"],
            ["black", "african", "F"],
            ["black", "african", "F"],
            ["black", "african", "F"],
            ["asian", "chinese", "F"],
            ["asian", "chinese", "F"],
            ["asian", "chinese", "F"],
            ["asian", "chinese", "F"],
            ["asian", "chinese", "F"]
        ], columns=SEED_FIELDS)

        model.generate(seed_df=seed_df)