from typing import List, Optional

from gretel_trainer.relational.connectors import (PostgreSQL, SQLite,
                                                  _Connection)
from gretel_trainer.relational.core import MultiTableException
from gretel_trainer.relational.multi_table import MultiTable


# This is a helper class that orchestrates an end-to-end workflow in Trainer.
#
# Long term, we anticipate connecters being Java components,
# so we want to make sure we have some seam between databases (from which we
# extract source data and to which we write synthetic data) and models
# (which train on extracted source data and produce synthetic data).
#
# For now, it's useful to ignore ser/de details and keep everything in Python.
# Inline pseudocode comments exist in the code below representing the boundaries
# between these components.
class MultiTableEndToEnd:
    def __init__(
        self,
        src_db_path: str,
        synth_record_size_ratio: float,
        dest_db_path: Optional[str] = None,
        tables_not_to_synthesize: Optional[List[str]] = None,
    ):
        if dest_db_path is None:
            dest_db_path = src_db_path
        self.src_connector = _make_connector(src_db_path)
        self.dest_connector = _make_connector(dest_db_path)
        self.synth_record_size_ratio = synth_record_size_ratio
        self.tables_not_to_synthesize = tables_not_to_synthesize

    def execute(self):
        relational_data = self.src_connector.extract()
        # source_metadata_path = relational_data.to_filesystem()

        # relational_data = RelationalData.from_filesystem(source_metadata_path)
        model = MultiTable(
            relational_data=relational_data,
        )
        model.train()
        synthetic_tables = model.generate(
            record_size_ratio=self.synth_record_size_ratio,
            preserve_tables=self.tables_not_to_synthesize,
        )
        # synthetic_metadata_path = write_synthetic_data_to_filesystem(synthetic_tables)

        # self.dest_connector.import(synthetic_metadata_path)  # would presumably replace the line below
        self.dest_connector.save_to_db(synthetic_tables)


def _make_connector(db_path: str) -> _Connection:
    if "sqlite://" in db_path:
        return SQLite(db_path)
    elif "postgres://" in db_path:
        return PostgreSQL(db_path)
    else:
        raise MultiTableException("Unrecognized db path string")
