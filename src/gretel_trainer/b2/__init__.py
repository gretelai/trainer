import logging

from gretel_trainer.b2.comparison import compare
from gretel_trainer.b2.core import Datatype
from gretel_trainer.b2.custom.datasets import make_dataset
from gretel_trainer.b2.gretel.datasets import GretelDatasetRepo
from gretel_trainer.b2.gretel.models import (
    GretelACTGAN,
    GretelAmplify,
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
logger = logging.getLogger("gretel_trainer.b2")
logger.handlers.clear()
logger.addHandler(handler)
logger.setLevel("INFO")
