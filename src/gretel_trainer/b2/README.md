# B2

A v2 rewrite of Benchmark.

## Changes/improvements

- v1 went overboard with dependency injection and protocols; v2 code is overall easier to follow
- v2 Gretel Datasets are much more useful; you could conceivably use the v2 `GretelDatasetRepo` to get datasets for non-Benchmark usage
- v2 Custom Datasets are more intuitive to create
  - instead of providing a list of sources, you just pass a single source (either a string path to CSV or a pd.DataFrame)
  - the v2 Datatype enum is simplified
    - str and enum formats match (e.g. `Datatype.natural_language == "natural_language"`)
    - v1 "tabular_mixed" and "tabular_numeric" consolidated in v2 to just a singular "tabular"
- User can decide if running Gretel models via SDK or Trainer
- SDK models use a smarter polling mechanism than the `poll` helper
- When using SDK, all runs are grouped in a single project
- Quieter, more informative log output


## CSVs over DataFrames

I've updated datasets to no longer carry the weight of pandas dataframes in memory;
instead, they read the dataframe to get the row and column counts and then drop the dataframe on the floor,
keeping just a pointer to a CSV (either local or in S3).

I wonder if we could do something similar for generated synthetic data.
Instead of carrying the synthetic data as a DF, write it to a CSV someplace and just keep the path pointer.
- Trainer: `generate` returns a pandas DF; we could write that to the working dir and store the path (maybe on the RunStatus object)
- SDK: data exists in the cloud, can be downloaded from record handler; we can do the same thing as above, write to working dir and store the path
- Custom: instruct users to write their synthetic data to CSV; we can pass in keyword args for them to use (e.g. working directory path, or maybe output filename to use)

To that last point above, maybe the custom `generate` signature should return a `Path`?


## TODOs

- Trainer model names seem inconsistent but I haven't identified the exact pattern yet. Can we clean it up?
- Enforce (?) Custom Dataset names adhere to Gretel model name restrictions
- CustomExecutor
- GretelAuto (+ validation check only present when trainer=True)
- GretelDGAN and GretelGPTX validations (check only present when trainer=False)
- Port over the (auto) cleanup stuff?
