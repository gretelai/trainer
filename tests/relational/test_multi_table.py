import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, call, patch

from gretel_trainer.relational.multi_table import (
    MultiTable,
    TableEvaluation,
    TrainStatus,
)


def test_training_through_trainer(pets, configured_session):
    with tempfile.TemporaryDirectory() as work_dir, patch(
        "gretel_trainer.trainer.create_or_get_unique_project"
    ) as create_or_get_unique_project, patch(
        "gretel_trainer.relational.multi_table.create_or_get_unique_project"
    ) as get_proj, patch(
        "gretel_trainer.trainer.Trainer.train"
    ) as train, patch(
        "gretel_trainer.trainer.Trainer.trained_successfully"
    ) as trained_successfully:
        mock_upload_artifact = Mock()
        mock_project = Mock()
        mock_project.upload_artifact = mock_upload_artifact
        get_proj.return_value = mock_project
        create_or_get_unique_project.return_value = mock_project

        trained_successfully.return_value = True
        multitable = MultiTable(pets, working_dir=work_dir)

        multitable.train()

        mock_upload_artifact.assert_called_once_with(
            f"{work_dir}/_gretel_debug_summary.json"
        )

        for table in ["humans", "pets"]:
            training_csv = Path(f"{work_dir}/{table}-train.csv")
            assert os.path.exists(training_csv)
            train.assert_any_call(training_csv)


def test_evaluate(source_nba, synthetic_nba, configured_session):
    rel_data, _, _, _ = source_nba
    _, syn_states, syn_cities, syn_teams = synthetic_nba

    with tempfile.TemporaryDirectory() as work_dir, patch(
        "gretel_trainer.relational.multi_table.create_or_get_unique_project"
    ) as create_or_get_unique_project:
        multitable = MultiTable(rel_data)

    with patch(
        "gretel_trainer.relational.multi_table.Trainer.load"
    ) as load_trainer, patch(
        "gretel_trainer.relational.multi_table.QualityReport"
    ) as quality_report:
        trainer = Mock()
        trainer.get_sqs_score.return_value = 42
        load_trainer.return_value = trainer

        report = quality_report.return_value
        report.run.return_value = None
        report.peek = lambda: {"score": 84}

        multitable.train_statuses["cities"] = TrainStatus.Completed

        evaluations = multitable.evaluate(
            {
                "states": syn_states,
                "cities": syn_cities,
                "teams": syn_teams,
            }
        )

    assert evaluations["states"] == TableEvaluation(individual_sqs=84, ancestral_sqs=84)
    assert evaluations["cities"] == TableEvaluation(individual_sqs=42, ancestral_sqs=84)
