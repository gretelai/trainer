# Gretel Trainer

This code is designed to help users successfully train synthetic models on complex datasets with high row and column counts. The code works by intelligently dividing a dataset into a set of smaller datasets of correlated columns that can be parallelized and then joined together.

# Install

**Using `pip`:**

```bash
pip install -U gretel-trainer
```

# Quickstart

1. Add your [Gretel API](https://console.gretel.cloud) key via the Gretel CLI.
```bash
gretel configure
```

2. Train or fine-tune a model using the Gretel API

```python3
from gretel_trainer import trainer

dataset = "https://gretel-public-website.s3-us-west-2.amazonaws.com/datasets/USAdultIncome5k.csv"

model = trainer.Trainer()
model.train(dataset)
```

3. Generate synthetic data! 
```python3
df = model.generate()
```

# TODOs / Roadmap

- [ ] Enable conditional generation via SDK interface (supported in Notebooks currently).
