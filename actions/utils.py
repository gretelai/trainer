"""
Common utils for action modules
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from gretel_client import configure_session
from gretel_client.projects.models import read_model_config
from gretel_trainer.relational.connectors import Connector
from pydantic import BaseSettings, SecretStr
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError


class GretelSettingsError(Exception):
    ...


class GretelSettings(BaseSettings):
    """
    Environment variables that will be read in on init. Depending
    on the action, a ".env" file should be created with the
    necessary variables set.
    """

    gretel_project: Optional[str]
    gretel_project_display_name: Optional[str]
    gretel_api_key: Optional[SecretStr]
    gretel_endpoint: Optional[str]
    gretel_config: Optional[str]
    gretel_runner: Optional[str]
    gretel_artifact_endpoint: Optional[str]
    gretel_work_dir: Optional[str]
    source_db: Optional[SecretStr]
    sink_db: Optional[SecretStr]

    class Config:
        env_file = ".env"


class ActionUtils:

    settings: GretelSettings

    def __init__(self):
        self.settings = GretelSettings()

    def validate_db_connections(self) -> None:
        errors = []
        for conn, desc in zip(
            (self.settings.source_db, self.settings.sink_db), ("source db", "sink db")
        ):
            if conn is None:
                errors.append(f"{conn} connection string is not set")

            engine = create_engine(conn.get_secret_value())
            try:
                engine.connect()
            except OperationalError as err:
                errors.append(f"{desc} failed with: {str(err)}")

        if errors:
            err_str = "The following errors occured: " + ", ".join(errors)
            raise GretelSettingsError(err_str)

    def connect_gretel(self) -> None:
        api_key = None
        if self.settings.gretel_api_key:
            api_key = self.settings.gretel_api_key.get_secret_value()

        configure_session(
            api_key=api_key,
            default_runner=self.settings.gretel_runner,
            artifact_endpoint=self.settings.gretel_artifact_endpoint,
            endpoint=self.settings.gretel_endpoint,
            validate=True,
        )

    @property
    def gretel_config(self) -> dict:
        return read_model_config(self.settings.gretel_config)

    @property
    def work_dir(self) -> Optional[str]:
        if not self.settings.gretel_work_dir:
            return None

        if not Path(self.settings.gretel_work_dir).is_dir():
            raise GretelSettingsError(
                f"{self.settings.gretel_work_dir} is not a directory"
            )

        return self.settings.gretel_work_dir

    def bootstrap(self) -> ActionUtils:
        """
        Run some checks to make sure that we can access all the
        resources and permissions we need.
        """
        self.connect_gretel()
        _ = self.gretel_config
        _ = self.work_dir
        self.validate_db_connections()
        return self

    def get_connector(self, conn_type: str) -> Connector:
        if conn_type not in ("source_db", "sink_db"):
            raise ValueError("invalid connection type")

        db_str: SecretStr = getattr(self.settings, conn_type, None)
        if not db_str:
            raise GretelSettingsError(
                f"There is no connection specified for: {conn_type}"
            )

        return Connector(create_engine(db_str.get_secret_value()))
