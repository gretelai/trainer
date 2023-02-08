import logging

# Some of these are imported simply to ensure the logger is instantiated before getting configured below
import gretel_trainer.relational.sdk_extras
import gretel_trainer.relational.strategies.ancestral
import gretel_trainer.relational.strategies.common
import gretel_trainer.relational.strategies.independent
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
    "gretel_trainer.relational.sdk_extras": "INFO",
    "gretel_trainer.relational.strategies.ancestral": "INFO",
    "gretel_trainer.relational.strategies.common": "INFO",
    "gretel_trainer.relational.strategies.independent": "INFO",
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


def create_report(multitable: MultiTable) -> None:
    logger = logging.getLogger("gretel_trainer.relational.multi_table")
    logger.info(
        "The `create_report` function is deprecated and will be removed in a future release. Instead call the `MultiTable#create_relational_report` instance method."
    )
    multitable.create_relational_report()
