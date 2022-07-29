"""Model specific parameters and definitions"""

from collections import namedtuple
from enum import Enum


class BaseConfig:
    """This class should not be used directly, model-specific configs should be
    derived from this class
    """

    def __init__(self, config_file: str, max_rows: int, max_header_clusters: int):
        self.config_file = config_file
        self.max_rows = max_rows
        self.max_header_clusters = max_header_clusters


class GretelLSTMConfig(BaseConfig):

    config = "synthetics/default"
    max_header_clusters = 20
    max_rows = 50000

    def __init__(self):
        super().__init__(
            config_file=self.config,
            max_rows=self.max_rows,
            max_header_clusters=self.max_header_clusters,
        )


class GretelCTGANConfig(BaseConfig):

    config = "synthetics/high-dimensionality"
    max_header_clusters = 100
    max_rows = 50000

    def __init__(self):
        super().__init__(
            config_file=self.config,
            max_rows=self.max_rows,
            max_header_clusters=self.max_header_clusters,
        )


class ExtendedEnum(Enum):
    """Utility class for Model enum"""

    @classmethod
    def get_types(cls):
        return list(map(lambda c: c.name, cls))

    @classmethod
    def get_config(cls, model: str) -> str:
        return cls[model].config.config_file

    @classmethod
    def get_max_rows(cls, model: str) -> int:
        return cls[model].config.max_rows

    @classmethod
    def get_max_header_clusters(cls, model: str) -> int:
        return cls[model].config.max_header_clusters


class Model(namedtuple("Model", "config"), ExtendedEnum):
    """Enum to pair default models and configurations"""

    GretelLSTM = GretelLSTMConfig()
    GretelCTGAN = GretelCTGANConfig()
