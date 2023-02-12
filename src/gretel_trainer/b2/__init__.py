import logging
from typing import List

from gretel_trainer.b2.comparison import Comparison, ModelTypes
from gretel_trainer.b2.core import BenchmarkConfig, Dataset, Datatype
from gretel_trainer.b2.custom_datasets import make_dataset
from gretel_trainer.b2.gretel_datasets import GretelDatasetRepo
from gretel_trainer.b2.gretel_models import (
    GretelAmplify,
    GretelACTGAN,
    GretelGPTX,
    GretelLSTM,
    GretelModel,
)

log_levels = {
    "gretel_trainer.b2.comparison": "INFO",
    "gretel_trainer.b2.gretel_executor": "INFO",
}

log_format = "%(levelname)s - %(asctime)s - %(message)s"
time_format = "%Y-%m-%d %H:%M:%S"

# Clear out any existing root handlers
# (This prevents duplicate log output in Colab)
for root_handler in logging.root.handlers:
    logging.root.removeHandler(root_handler)

# Configure our loggers
for name, level in log_levels.items():
    logger = logging.getLogger(name)

    formatter = logging.Formatter(log_format, time_format)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)


def compare(
    *,
    datasets: List[Dataset],
    models: List[ModelTypes],
    project_display_name: str = "benchmark",
    trainer: bool = False,
    refresh_interval: int = 15,
) -> Comparison:
    config = BenchmarkConfig(
        project_display_name=project_display_name,
        trainer=trainer,
        refresh_interval=refresh_interval,
    )
    comparison = Comparison(
        datasets=datasets,
        models=models,
        config=config,
    )
    return comparison.execute()

