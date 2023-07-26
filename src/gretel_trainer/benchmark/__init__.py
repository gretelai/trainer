import logging

from gretel_trainer.benchmark.core import BenchmarkConfig, Datatype
from gretel_trainer.benchmark.custom.datasets import create_dataset, make_dataset
from gretel_trainer.benchmark.entrypoints import compare, launch
from gretel_trainer.benchmark.gretel.datasets import GretelDatasetRepo
from gretel_trainer.benchmark.gretel.datasets_backwards_compatibility import (
    get_gretel_dataset,
    list_gretel_dataset_tags,
    list_gretel_datasets,
)
from gretel_trainer.benchmark.gretel.models import (
    GretelACTGAN,
    GretelAmplify,
    GretelAuto,
    GretelDGAN,
    GretelGPTX,
    GretelLSTM,
    GretelModel,
)

log_format = "%(levelname)s - %(asctime)s - %(message)s"
time_format = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(log_format, time_format)
handler = logging.StreamHandler()
handler.setFormatter(formatter)

# Clear out any existing root handlers
# (This prevents duplicate log output in Colab)
for root_handler in logging.root.handlers:
    logging.root.removeHandler(root_handler)

# Configure benchmark loggers
logger = logging.getLogger("gretel_trainer.benchmark")
logger.handlers.clear()
logger.addHandler(handler)
logger.setLevel("INFO")
