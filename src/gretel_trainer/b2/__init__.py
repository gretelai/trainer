import logging

from gretel_trainer.b2.comparison import compare
from gretel_trainer.b2.core import Datatype
from gretel_trainer.b2.custom_datasets import make_dataset
from gretel_trainer.b2.gretel_datasets import GretelDatasetRepo
from gretel_trainer.b2.gretel_models import (
    GretelACTGAN,
    GretelAmplify,
    GretelGPTX,
    GretelLSTM,
    GretelModel,
)

log_levels = {
    "gretel_trainer.b2.comparison": "INFO",
    "gretel_trainer.b2.custom_executor": "INFO",
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
