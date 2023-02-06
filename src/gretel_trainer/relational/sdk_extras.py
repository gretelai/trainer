from typing import Dict

from gretel_client.projects.jobs import Job, Status


def cautiously_refresh_status(
    job: Job, key: str, refresh_attempts: Dict[str, int]
) -> Status:
    try:
        job.refresh()
        refresh_attempts[key] = 0
    except:
        refresh_attempts[key] = refresh_attempts[key] + 1

    return job.status
