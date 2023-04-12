import logging

RELATIONAL = "gretel_trainer.relational"

log_format = "%(levelname)s - %(asctime)s - %(message)s"
time_format = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(log_format, time_format)
handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger = logging.getLogger(RELATIONAL)
logger.handlers.clear()
logger.addHandler(handler)
logger.setLevel("INFO")

# Clear out any existing root handlers
# (This prevents duplicate log output in Colab)
for root_handler in logging.root.handlers:
    logging.root.removeHandler(root_handler)
