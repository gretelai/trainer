# Gretel Trainer

This code is designed to help users successfully train synthetic models on complex datasets with high row and column counts. The code works by intelligently dividing a dataset into a set of smaller datasets of correlated columns that can be parallelized and then joined together.

# Get Started

## Running the notebook
Launch the [Notebook](https://github.com/gretelai/trainer/blob/main/notebooks/gretel-trainer.ipynb) in [Google Colab](https://colab.research.google.com/github/gretelai/trainer/blob/main/notebooks/gretel-trainer.ipynb) or your preferred environment. 

**NOTE**: Either delete the existing or choose a new cache file name if you are starting
a dataset run from scratch.

# TODOs / Roadmap

- [ ] Enable additional sampling from from trained models.
- [ ] Detect and label encode random UIDs (preprocessing).
