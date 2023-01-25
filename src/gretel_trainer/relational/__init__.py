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
from gretel_trainer.relational.report.report import create_report

# Optimize logging for multitable output
log_levels = {
    "gretel_trainer.relational.connectors": "INFO",
    "gretel_trainer.relational.core": "INFO",
    "gretel_trainer.relational.multi_table": "INFO",
    "gretel_trainer.relational.strategies.common": "INFO",
}

log_format = "%(levelname)s - %(asctime)s - %(message)s"
time_format = "%Y-%m-%d %H:%M:%S"

# Clear out any existing root handlers
# (This prevents duplicate log output in Colab)
for root_handler in logging.root.handlers:
    logging.root.removeHandler(root_handler)

# Configure our loggers
for name, level in log_levels.items():
    logger = logging.getLogger(name)

    formatter = logging.Formatter(log_format, time_format)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)
