import logging
from typing import Optional

import pandas as pd
from gretel_client.projects.models import Model
from gretel_client.projects.projects import Project
from gretel_client.projects.records import RecordHandler

from gretel_trainer.b2.core import BenchmarkException, Dataset
from gretel_trainer.b2.gretel_models import GretelModel

logger = logging.getLogger(__name__)


class GretelExecutor:
    def __init__(self, project: Project, benchmark_model: GretelModel):
        self.project = project
        self.benchmark_model = benchmark_model

        self.model: Optional[Model] = None
        self.source: Optional[Dataset] = None
        self.record_handler: Optional[RecordHandler] = None

    def train(self, source: Dataset) -> None:
        logger.info(f"Starting training of {self.benchmark_model.name} on {source.name}")
        self.source = source
        self.model = self.project.create_model_obj(
            model_config=self.benchmark_model.config, data_source=source
        )
        self.model.submit_cloud()

    def generate(self) -> None:
        if self.model is None or self.source is None:
            raise BenchmarkException("Cannot generate before training")

        logger.info(f"Starting synthetic data generation for {self.benchmark_model.name} trained on {self.source.name}")
        self.record_handler = self.model.create_record_handler_obj(
            params={"num_records": self.source.row_count}
        )
        self.record_handler.submit_cloud()

    def get_sqs_score(self) -> int:
        if self.model is None:
            raise BenchmarkException("Cannot get SQS score before training")

        report = self.model.peek_report()
        return report["synthetic_data_quality_score"]["score"]

    def get_synthetic_data(self) -> pd.DataFrame:
        if self.record_handler is None:
            raise BenchmarkException("Cannot get synthetic data before generating")

        return pd.read_csv(self.record_handler.get_artifact_link("data"), compression="gzip")
