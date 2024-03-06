import logging

import gretel_trainer.relational.log

from gretel_trainer.relational.connectors import (
    Connector,
    mariadb_conn,
    mysql_conn,
    postgres_conn,
    snowflake_conn,
    sqlite_conn,
)
from gretel_trainer.relational.core import RelationalData
from gretel_trainer.relational.extractor import ExtractorConfig
from gretel_trainer.relational.log import set_log_level
from gretel_trainer.relational.multi_table import MultiTable

logger = logging.getLogger(__name__)

logger.warn(
    "Relational Trainer is deprecated, and will be removed in the next Trainer release. "
    "To transform and synthesize relational data, use Gretel Workflows. "
    "Visit the docs to learn more: https://docs.gretel.ai/create-synthetic-data/workflows-and-connectors"
)
