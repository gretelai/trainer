# Relational

## Quickstart

```python
from gretel_trainer.relational.connectors import sqlite_conn
from gretel_trainer.relational.multi_table import MultiTable

connection = sqlite_conn(path="my_data.db")
relational_data = connection.extract()
multitable = MultiTable(relational_data)
multitable.train()
synthetic_tables = multitable.generate()
```

## Overview of Python objects

### `RelationalData`

This core data structure holds individual table data and table relationship metadata.
It can be created and modified manually, or can be extracted from a database via a
`Connector` instance.


### `Connector`

A `Connector` automates the extraction of table data and metadata from a database to
a `RelationalData` instance. A handful of helper functions exist to create `Connector`
instances for some popular databases, but this set is not exhaustive—your mileage may
vary depending on complex native database types, but generally speaking if you can
create a [SQLAlchemy Engine](https://docs.sqlalchemy.org/en/14/core/engines.html) for
your database, you can use the `Connector` class. Note these typically require local
installation of database drivers, for example `psycopg2-binary` for PostgreSQL,
`snowflake-sqlalchemy` for Snowflake, etc. These requirements are *not* bundled with
Gretel Trainer.


### `MultiTable`

This is the interface to working with `RelationalData`, including running transforms
on it, training synthetic models on it and generating data from those models, and
evaluating synthetic results.
