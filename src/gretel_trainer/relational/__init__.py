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

log_format = "%(levelname)s - %(asctime)s - %(message)s"
time_format = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(log_format, time_format)
handler = logging.StreamHandler()
handler.setFormatter(formatter)

# Clear out any existing root handlers
# (This prevents duplicate log output in Colab)
for root_handler in logging.root.handlers:
    logging.root.removeHandler(root_handler)

# Configure relational loggers
logger = logging.getLogger("gretel_trainer.relational")
logger.handlers.clear()
logger.addHandler(handler)
logger.setLevel("INFO")
