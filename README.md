# Gretel Trainer

This module is designed to provide a simple interface to help users successfully train synthetic models on complex datasets with high row and column counts, and offers features such as Cloud SaaS based training and multi-GPU based parallelization. Get started for free with an API key from [Gretel.ai](https://console.gretel.cloud).

## Current functionality and features:

* Synthetic data generators for text, tabular, and time-series data with the following
  features:
    * Balance datasets or boost a minority class using Conditional Data Generation.
    * Automated data validation.
    * Synthetic data quality reports.
    * Privacy filters and optional differential privacy support.
* Multiple [model types supported](https://docs.gretel.ai/synthetics/models):
    * `Gretel-LSTM` model type supports text, tabular, time-series, and conditional data generation.
    * `Gretel-ACTGAN` model type supports tabular and conditional data generation.
    * `Gretel-GPT` natural language synthesis based on an open-source implementation of GPT-3 (coming soon).
    * `Gretel-DGAN` multi-variate time series based on DoppelGANger (coming soon).

## Try it out now!

If you want to quickly get started synthesizing data with **Gretel.ai**, simply click the button below and follow the examples. See additional Python3 and Jupyter Notebook examples in the `./notebooks` folder.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gretelai/trainer/blob/main/notebooks/trainer-examples.ipynb)

## Join the Synthetic Data Community Discord

If you want to be part of the Synthetic Data Community to receive announcements of the latest releases,
ask questions, suggest new features or participate in the development meetings, please join
the Synthetic Data Community Server!

[![Discord](https://img.shields.io/discord/1007817822614847500?label=Discord&logo=Discord)](https://gretel.ai/discord)

# Install

**Using `pip`:**

```bash
pip install -U gretel-trainer
```

# Quickstart

## 1. Add your [Gretel API](https://console.gretel.cloud) key via the Gretel CLI.
Use the Gretel client to store your API key to disk. This step is optional, the trainer will prompt for an API key in the next step.
```bash
gretel configure
```

## 2. Train or fine-tune a model using the Gretel API

```python3
from gretel_trainer import trainer

dataset = "https://gretel-public-website.s3-us-west-2.amazonaws.com/datasets/USAdultIncome5k.csv"

model = trainer.Trainer()
model.train(dataset)
```

## 3. Generate synthetic data!
```python3
df = model.generate()
```

## Development

- Run tests via `make test`
- Run type-checking (limited coverage) via `make type`

## TODOs / Roadmap

- [ ] Enable conditional generation via SDK interface (supported in Notebooks currently).
