import logging

from gretel_trainer.relational.connectors import (
    Connector,
    mariadb_conn,
    mysql_conn,
    postgres_conn,
    snowflake_conn,
    sqlite_conn,
)
from gretel_trainer.relational.core import RelationalData
from gretel_trainer.relational.multi_table import MultiTable

# Optimize logging for multitable output
log_levels = {
    "gretel_trainer.relational.connectors": "INFO",
    "gretel_trainer.relational.core": "INFO",
    "gretel_trainer.relational.multi_table": "INFO",
}

for name, level in log_levels.items():
    logging.getLogger(name).setLevel(level)
