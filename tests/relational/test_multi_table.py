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

        humans_training_csv = Path(f"{work_dir}/humans.csv")
        assert os.path.exists(humans_training_csv)
        train.assert_any_call(humans_training_csv, seed_fields=None)

        pets_training_csv = Path(f"{work_dir}/pets.csv")
        assert os.path.exists(pets_training_csv)
        train.assert_any_call(
            pets_training_csv,
            seed_fields=[
                "self.human_id|name",
                "self.human_id|city",
                "self.human_id|id",
            ],
        )
