import logging

BENCHMARK = "gretel_trainer.benchmark"

log_format = "%(levelname)s - %(asctime)s - %(message)s"
time_format = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(log_format, time_format)
handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger = logging.getLogger(BENCHMARK)
logger.handlers.clear()
logger.addHandler(handler)
logger.setLevel("INFO")

# Clear out any existing root handlers
# (This prevents duplicate log output in Colab)
for root_handler in logging.root.handlers:
    logging.root.removeHandler(root_handler)


def set_log_level(level: str):
    logger = logging.getLogger(BENCHMARK)
    logger.setLevel(level)
