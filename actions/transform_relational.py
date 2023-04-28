"""
Transform a Database
"""
import copy
from typing import Optional

from gretel_trainer.relational.multi_table import MultiTable

from utils import ActionUtils


def transform_db(*, action_utils: Optional[ActionUtils] = None):
    if action_utils is None:
        action_utils = ActionUtils()
    action_utils.bootstrap()
    source_db_conn = action_utils.get_connector("source_db")
    source_relational_data = source_db_conn.extract()
    multi_table = MultiTable(
        relational_data=source_relational_data,
        project_display_name=action_utils.settings.gretel_project_display_name,
        # TODO: enable this once this param is available on the class
        # base_work_dir=action_utils.work_dir
    )
    gretel_config = action_utils.gretel_config

    # For this action, we just use one Gretel Config to
    # transform every table in the DB
    configs = {}
    all_tables = source_relational_data.list_all_tables()
    for table_name in all_tables:
        configs[table_name] = copy.deepcopy(gretel_config)

    action_utils.send_webhook(f"Creating Gretel Transform Models for {len(all_tables)} tables from the configured database.")
    multi_table.train_transform_models(configs=configs)
    action_utils.send_webhook(f"Finished training Gretel Transform Models, transforming {len(all_tables)} database tables.")
    multi_table.run_transforms()
    action_utils.send_webhook("Database transform complete, writing to sink database.")
    sink_db_conn = action_utils.get_connector("sink_db")
    sink_db_conn.save(multi_table.transform_output_tables)


if __name__ == "__main__":
    transform_db()
