"""
Generic entrypoint that can read a specific action from the
environment and start the work.
"""
import sys
import logging

from utils import ActionUtils

from transform_relational import transform_db

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ACTIONS = {
    "transform_relational": (
        "Transform values in a relational database with a Gretel Transform Configuration",
        transform_db,
    )
}

if __name__ == "__main__":
    action_utils = ActionUtils()
    action_name = action_utils.settings.gretel_action
    if not action_name:
        logger.error("No 'GRETEL_ACTION' was found in the environment, quitting.")
        sys.exit(1)

    action_tuple = ACTIONS.get(action_name)

    if not action_tuple:
        logger.error(
            f"Invalid GRETEL_ACTION, possible values are: {', '.join(list(ACTIONS.keys()))}, received: {action_name}"
        )
        sys.exit(1)

    _, action_fn = action_tuple
    logger.info(f"Starting action: {action_name}")
    action_fn()
