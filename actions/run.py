"""
Generic entrypoint that can read a specific action from the
environment and start the work.
"""
import logging
import sys

from transform_relational import transform_db
from utils import ActionUtils

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
    action_utils.send_webhook(f"Starting Gretel Trainer Action: {action_name} for Gretel user: {action_utils.this_gretel_user}")
    logger.info(f"Starting action: {action_name} (Action ID: {action_utils.action_id})")
    try:
        action_fn(action_utils=action_utils)
    except Exception as err:
        action_utils.send_webhook(f"The Gretel Trainer Action: {action_name} encountered an error: {str(err)}")
        raise
    else:
        action_utils.send_webhook(f"Gretel Trainer Action {action_name} has completed.")
