from typing import List, Optional, Type, Union

import pandas as pd
from gretel_client import configure_session
from gretel_client.projects import create_project

from gretel_trainer.b2.core import BenchmarkConfig, Dataset
from gretel_trainer.b2.custom_models import CustomModel
from gretel_trainer.b2.gretel_executor import GretelExecutor
from gretel_trainer.b2.gretel_models import GretelModel


ModelTypes = Union[Type[CustomModel], Type[GretelModel]]


class Comparison:
    def __init__(
        self,
        *,
        datasets: List[Dataset],
        models: List[ModelTypes],
        config: Optional[BenchmarkConfig] = None,
    ):
        self.datasets = datasets
        self.gretel_models = [m for m in models if issubclass(m, GretelModel)]
        self.custom_models = [m for m in models if not m in self.gretel_models]
        self.config = config or BenchmarkConfig()

        configure_session(api_key="prompt", cache="yes", validate=True)
        self._project = create_project(display_name=self.config.project_display_name)

    def execute(self):
        for dataset in self.datasets:
            for model in self.gretel_models:
                exector = GretelExecutor(project=self._project, benchmark_model=model())
                # kick off train and execute!
            for model in self.custom_models:
                pass
