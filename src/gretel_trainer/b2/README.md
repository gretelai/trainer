# B2

A v2 rewrite of Benchmark.


## User-facing interface changes

- Custom datasets `make_dataset`
  - Accepts single source instead of a list
  - Requires a user-specified name
  - Drops namespace
- `Datatype` enum
  - v1 "tabular_mixed" and "tabular_numeric" have been consolidated to just "tabular"
  - enum variant casing is lower-cased to more closely match string format
- Gretel dataset functions are no longer "free-standing" but rather methods on a `GretelDatasetRepo` object
- When running with SDK, all runs are grouped in a single project
- Auto-cleanup replaced by explicit `clean` function that must be manually called


## Benefits

- Code is easier to follow, less indirection
- Significant performance improvements
- Log output is higher level and more useful
- v2 Gretel Datasets are much more useful; you could conceivably use the v2 `GretelDatasetRepo` to get datasets for non-Benchmark usage
- v2 Custom Datasets are more intuitive to create
- User can choose explicitly whether to run Trainer or standard SDK for all models
- Generated synthetic data is written to the working directory so it is immediately accessible/reviewable


## TODOs

- Consider being more economical about data sources. Example: SDK comparisons can upload datasets' data sources in advance and pass the artifact keys to the SDKStrategy. We may need an artifact eviction strategy.
- Errors should be more detailed. Might help to do `raise BenchmarkException(...) from e` in certain places. Also some of those exceptions can get more detailed messages (e.g. exact job status)
- Is it simple enough to get to an error? `comparison.executors[run_id].exception`. Maybe some kind of `all_errors` helper function? With advice on what to do next (e.g. check logs)?
- Project name(s). Consider creating projects with both a display name (provided by user) **and** an actual unique name (something like `benchmark-{timestamp}`).
- Nicole: downstream evaluations
- Sami: resume functionality
- Sami: skip SQS and data preview when training, and instead always run Evaluate afterwards (improves training time + is more accurate for Trainer models than average SQS from N models)
- Sami: suggestion to not chain `compare->execute`. Not sure how I feel about this one. Maybe we keep `compare` the way it is but make it easier for people to instantiate a `Comparison` explicitly?
- Maarten: cancel/abort/stop as much as possible
- Maarten: hyperparam sweeping
