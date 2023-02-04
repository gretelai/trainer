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
