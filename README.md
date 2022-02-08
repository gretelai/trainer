# Gretel Trainer

This code is designed to help users successfully train synthetic models on complex datasets with high row and column counts. The code works by intelligently dividing a dataset into a set of smaller datasets of correlated columns that can be parallelized and then joined together.

# Getting Started

Make sure you have the latest `gretel-client` and `gretel-synthetics` installed. Run `gretel configure` from the command line to cache API credentials, and give the Notebook a try.

**NOTE**: Either delete the existing or choose a new cache file name if you are starting
a dataset run from scratch.

# TODOs / Roadmap

- [ ] Enable additional sampling from from trained models.
- [ ] Detect and label encode random UIDs (preprocessing).
- [ ] How to further partition model creation jobs that fail after retries?
- [x] If a single model fails, how many times to retry?
  - Default is set to 3, configurable as a param on the runner class
