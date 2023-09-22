import json
import logging

from pathlib import Path
from typing import Optional

import smart_open

import gretel_trainer.relational.tasks.common as common

from gretel_client.projects.jobs import Job
from gretel_client.projects.models import Model
from gretel_client.projects.projects import Project
from gretel_trainer.relational.sdk_extras import ExtendedGretelSDK
from gretel_trainer.relational.table_evaluation import TableEvaluation

logger = logging.getLogger(__name__)

ACTION = "synthetic data evaluation"


class SyntheticsEvaluateTask:
    def __init__(
        self,
        individual_evaluate_models: dict[str, Model],
        cross_table_evaluate_models: dict[str, Model],
        project: Project,
        run_dir: Path,
        evaluations: dict[str, TableEvaluation],
        multitable: common._MultiTable,
    ):
        self.jobs = {}
        for table, model in individual_evaluate_models.items():
            self.jobs[f"individual-{table}"] = model
        for table, model in cross_table_evaluate_models.items():
            self.jobs[f"cross_table-{table}"] = model
        self.project = project
        self.run_dir = run_dir
        self.evaluations = evaluations
        self.multitable = multitable
        self.completed = []
        self.failed = []

    def action(self, job: Job) -> str:
        return ACTION

    @property
    def table_collection(self) -> list[str]:
        return list(self.jobs.keys())

    @property
    def artifacts_per_job(self) -> int:
        return 2

    def more_to_do(self) -> bool:
        return len(self.completed + self.failed) < len(self.jobs)

    def wait(self) -> None:
        common.wait(self.multitable._refresh_interval)

    def is_finished(self, table: str) -> bool:
        return table in (self.completed + self.failed)

    def get_job(self, table: str) -> Job:
        return self.jobs[table]

    def handle_completed(self, table: str, job: Job) -> None:
        self.completed.append(table)
        common.log_success(table, ACTION)

        model = self.get_job(table)
        if table.startswith("individual-"):
            table_name = table.removeprefix("individual-")
            out_filepath = (
                self.run_dir / f"synthetics_individual_evaluation_{table_name}"
            )
            data = _get_reports(model, out_filepath, self.multitable._extended_sdk)
            self.evaluations[table_name].individual_report_json = data
        else:
            table_name = table.removeprefix("cross_table-")
            out_filepath = (
                self.run_dir / f"synthetics_cross_table_evaluation_{table_name}"
            )
            data = _get_reports(model, out_filepath, self.multitable._extended_sdk)
            self.evaluations[table_name].cross_table_report_json = data

        common.cleanup(sdk=self.multitable._extended_sdk, project=self.project, job=job)

    def handle_failed(self, table: str, job: Job) -> None:
        self.failed.append(table)
        common.log_failed(table, ACTION)
        common.cleanup(sdk=self.multitable._extended_sdk, project=self.project, job=job)

    def handle_lost_contact(self, table: str, job: Job) -> None:
        self.failed.append(table)
        common.log_lost_contact(table)
        common.cleanup(sdk=self.multitable._extended_sdk, project=self.project, job=job)

    def handle_in_progress(self, table: str, job: Job) -> None:
        common.log_in_progress(table, job.status, ACTION)

    def each_iteration(self) -> None:
        pass


def _get_reports(
    model: Model, out_filepath: Path, extended_sdk: ExtendedGretelSDK
) -> Optional[dict]:
    _download_reports(model, out_filepath, extended_sdk)
    return _read_json_report(model, out_filepath)


def _download_reports(
    model: Model, out_filepath: Path, extended_sdk: ExtendedGretelSDK
) -> None:
    """
    Downloads model reports to the provided path.
    """
    legend = {"html": "report", "json": "report_json"}

    for filetype, artifact_name in legend.items():
        out_path = f"{out_filepath}.{filetype}"
        extended_sdk.download_file_artifact(model, artifact_name, out_path)


def _read_json_report(model: Model, out_filepath: Path) -> Optional[dict]:
    """
    Reads the JSON report data in to a dictionary to be appended to the MultiTable
    evaluations property. First try reading the file we just downloaded to the run
    directory. If that fails, try reading the data remotely from the model. If that
    also fails, log a warning and give up gracefully.
    """
    full_path = f"{out_filepath}.json"
    try:
        return json.loads(smart_open.open(full_path).read())
    except:
        try:
            with model.get_artifact_handle("report_json") as report:
                return json.loads(report.read())
        except:
            logger.warning("Failed to fetch model evaluation report JSON.")
            return None
