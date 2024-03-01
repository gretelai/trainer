import json
import logging

from collections import defaultdict
from typing import Optional

import gretel_trainer.relational.tasks.common as common

from gretel_client.projects.artifact_handlers import open_artifact
from gretel_client.projects.jobs import Job
from gretel_client.projects.models import Model
from gretel_trainer.relational.output_handler import OutputHandler
from gretel_trainer.relational.table_evaluation import TableEvaluation
from gretel_trainer.relational.task_runner import TaskContext

logger = logging.getLogger(__name__)

ACTION = "synthetic data evaluation"


class SyntheticsEvaluateTask:
    def __init__(
        self,
        individual_evaluate_models: dict[str, Model],
        cross_table_evaluate_models: dict[str, Model],
        subdir: str,
        output_handler: OutputHandler,
        evaluations: dict[str, TableEvaluation],
        ctx: TaskContext,
    ):
        self.jobs = {}
        for table, model in individual_evaluate_models.items():
            self.jobs[f"individual-{table}"] = model
        for table, model in cross_table_evaluate_models.items():
            self.jobs[f"cross_table-{table}"] = model
        self.subdir = subdir
        self.output_handler = output_handler
        self.evaluations = evaluations
        self.completed = []
        self.failed = []
        self.ctx = ctx
        # Nested dict organizing by table > sqs_type > file_type, e.g.
        # {
        #     "users": {
        #         "individual": {
        #             "json": "/path/to/report.json",
        #             "html": "/path/to/report.html",
        #         },
        #         "cross_table": {
        #             "json": "/path/to/report.json",
        #             "html": "/path/to/report.html",
        #         },
        #     },
        # }
        self.report_filepaths: dict[str, dict[str, dict[str, str]]] = defaultdict(
            lambda: defaultdict(dict)
        )

    def action(self, job: Job) -> str:
        return ACTION

    @property
    def table_collection(self) -> list[str]:
        return list(self.jobs.keys())

    def more_to_do(self) -> bool:
        return len(self.completed + self.failed) < len(self.jobs)

    def is_finished(self, table: str) -> bool:
        return table in (self.completed + self.failed)

    def get_job(self, table: str) -> Job:
        return self.jobs[table]

    def handle_completed(self, table: str, job: Job) -> None:
        self.completed.append(table)
        common.log_success(table, self.action(job))

        model = self.get_job(table)
        sqs_type, table_name = table.split("-", 1)

        filename_stem = _filename_stem(sqs_type, table_name)

        # JSON
        json_filepath = self.output_handler.filepath_for(
            f"{filename_stem}.json", subdir=self.subdir
        )
        json_ok = self.ctx.extended_sdk.download_file_artifact(
            model, "report_json", json_filepath
        )
        if json_ok:
            self.report_filepaths[table_name][sqs_type]["json"] = json_filepath
        # Set json data on local evaluations object for use in report
        json_data = _read_json_report(model, json_filepath)
        if sqs_type == "individual":
            self.evaluations[table_name].individual_report_json = json_data
        else:
            self.evaluations[table_name].cross_table_report_json = json_data

        # HTML
        html_filepath = self.output_handler.filepath_for(
            f"{filename_stem}.html", subdir=self.subdir
        )
        html_ok = self.ctx.extended_sdk.download_file_artifact(
            model, "report", html_filepath
        )
        if html_ok:
            self.report_filepaths[table_name][sqs_type]["html"] = html_filepath

        common.cleanup(sdk=self.ctx.extended_sdk, project=self.ctx.project, job=job)

    def handle_failed(self, table: str, job: Job) -> None:
        self.failed.append(table)
        common.log_failed(table, self.action(job))
        common.cleanup(sdk=self.ctx.extended_sdk, project=self.ctx.project, job=job)

    def handle_lost_contact(self, table: str, job: Job) -> None:
        self.failed.append(table)
        common.log_lost_contact(table)
        common.cleanup(sdk=self.ctx.extended_sdk, project=self.ctx.project, job=job)

    def handle_in_progress(self, table: str, job: Job) -> None:
        common.log_in_progress(table, job.status, self.action(job))

    def each_iteration(self) -> None:
        pass


def _read_json_report(model: Model, json_report_filepath: str) -> Optional[dict]:
    """
    Reads the JSON report data in to a dictionary to be appended to the MultiTable
    evaluations property. First try reading the file we just downloaded to the run
    directory. If that fails, try reading the data remotely from the model. If that
    also fails, log a warning and give up gracefully.
    """
    try:
        return json.loads(open_artifact(json_report_filepath).read())
    except:
        try:
            with model.get_artifact_handle("report_json") as report:
                return json.loads(report.read())
        except:
            logger.warning("Failed to fetch model evaluation report JSON.")
            return None


def _filename_stem(sqs_type: str, table_name: str) -> str:
    return f"synthetics_{sqs_type}_evaluation_{table_name}"
