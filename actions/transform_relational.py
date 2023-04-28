"""
Transform a Database
"""
import copy

from gretel_trainer.relational.multi_table import MultiTable

from utils import ActionUtils


def transform_db():
    action_utils = ActionUtils().bootstrap()
    source_db_conn = action_utils.get_connector("source_db")
    source_relational_data = source_db_conn.extract()
    multi_table = MultiTable(
        relational_data=source_relational_data,
        project_display_name=action_utils.settings.gretel_project_display_name,
    )
    gretel_config = action_utils.gretel_config

    # For this action, we just use one Gretel Config to
    # transform every table in the DB
    configs = {}
    for table_name in source_relational_data.list_all_tables():
        configs[table_name] = copy.deepcopy(gretel_config)

    multi_table.train_transform_models(configs=configs)
    multi_table.run_transforms()
    sink_db_conn = action_utils.get_connector("sink_db")
    sink_db_conn.save(multi_table.transform_output_tables)


if __name__ == "__main__":
    transform_db()
