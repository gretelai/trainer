# Relational

## User guide

See `notebooks/multitable.ipynb` for a walkthrough of functionality.


## Overview of Python objects

### `RelationalData`

This core data structure holds individual table data and table relationship metadata.
It can be created and modified manually, or can be extracted from a database via a
`Connector` instance.


### `Connector`

A `Connector` automates the extraction of table data and metadata from a database to
a `RelationalData` instance. `Connector`s can also write synthetic data to a database.

A handful of helper functions exist to create `Connector` instances for some popular
databases, but this set is not exhaustive—your mileage may vary depending on complex
native database types, but generally speaking if you can create a
[SQLAlchemy Engine](https://docs.sqlalchemy.org/en/14/core/engines.html) for your
database, you can use the `Connector` class. Engines almost always require local
installation of database drivers like `psycopg2-binary` (PostgreSQL) or `mysqlclient`
(MySQL/MariaDB). The optional `[relational]` installation package installs the
drivers necessary for all the helper functions mentioned above. For more information,
see the [SQLAlchemy dialects docs](https://docs.sqlalchemy.org/en/14/dialects/index.html).


### `MultiTable`

This is the interface to working with `RelationalData`, including running transforms
on it, training synthetic models on it and generating data from those models, and
evaluating synthetic results.

The `MultiTable` object is primarily concerned with the orchestration, execution, and
state-tracking of Gretel Cloud jobs. While it does also contain some "business logic,"
most of those details are defined in the `Strategy` protocol. A `MultiTable` instance
depends on one `Strategy`.


### `Strategy` (protocol)

`Strategy` instances capture low-level business logic, including but not limited to:
- how source data is modified prior to training
- sequencing of table generation jobs
- post-processing of synthetic results from generation jobs

There are currently two strategies:
- `SingleTableStrategy`, in which models are trained on individual tables
- `CrossTableStrategy`, in which models are trained on multi-generational data via joined tables
