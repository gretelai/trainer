# Synthetic Job Partitioning

Utilities for automating job parallelization for big datasets.

# Refs

Le [MFR](https://gretel.atlassian.net/jira/software/projects/MFR/boards/16/roadmap?selectedIssue=MFR-126)

# Getting Started

Make sure you have the latest `gretel-client` and `gretel-synthetics` installed, and give the Notebook a try.

**NOTE**: Either delete the existing or choose a new cache file name if you are starting
a dataset run from scratch.

# TODOs / Roadmap

- [ ] Still need to create jobs that generate data and re-assesemle the syn DF
- [ ] How to further partition model creation jobs that fail after retries?
- [x] If a single model fails, how many times to retry?
  - Default is set to 3, configurable as a param on the runner class