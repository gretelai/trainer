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
- CSVs as contract instead of DataFrames
  - Significantly improves performance!
  - Generated data is written as CSV to the working directory so that it is accessible/reviewable


## TODOs

- Sami: resume functionality
- Sami: skip SQS and data preview when training, and instead always run Evaluate afterwards
  - Improves training time + is more accurate for Trainer models than average SQS from N models
- Sami: suggestion to not chain `compare->execute`. Not sure how I feel about this one. Maybe we keep `compare` the way it is but make it easier for people to instantiate a `Comparison` explicitly?
- Trainer model names seem inconsistent but I haven't identified the exact pattern yet. Can we clean it up?
- Enforce (?) Custom Dataset names adhere to Gretel model name restrictions
    - Maybe flip Run Identifier from "dset-model" to "model-dset" to avoid a short-dset name causing a "too-early" hyphen
- Port over the (auto) cleanup stuff?
