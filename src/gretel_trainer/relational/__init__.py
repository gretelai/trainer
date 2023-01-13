import logging

import gretel_trainer.relational.strategies.common
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
    "gretel_trainer.relational.strategies.common": "INFO",
}

log_format = "%(levelname)s - %(name)s - %(message)s"

for name, level in log_levels.items():
    logger = logging.getLogger(name)

    formatter = logging.Formatter(log_format)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)
