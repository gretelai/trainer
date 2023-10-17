import json

from collections import defaultdict
from unittest.mock import Mock

from gretel_trainer.relational.table_evaluation import TableEvaluation
from gretel_trainer.relational.tasks import SyntheticsEvaluateTask


def test_sets_json_data_on_evaluations(output_handler, project):
    json_result = {"sqs": 99}

    def mock_download_file_artifact(gretel_object, artifact_name, out_path):
        if artifact_name == "report_json":
            with open(out_path, "w") as out:
                json.dump(json_result, out)

    ind_users_model = Mock()
    evaluations = defaultdict(lambda: TableEvaluation())
    ext_sdk = Mock()
    ext_sdk.download_file_artifact.side_effect = mock_download_file_artifact
    multitable = Mock(_extended_sdk=ext_sdk)

    task = SyntheticsEvaluateTask(
        individual_evaluate_models={"users": ind_users_model},
        cross_table_evaluate_models={},
        project=project,
        subdir="run",
        output_handler=output_handler,
        evaluations=evaluations,
        multitable=multitable,
    )

    output_handler.make_subdirectory("run")
    task.handle_completed(table="individual-users", job=ind_users_model)

    assert evaluations["users"].individual_report_json == json_result
