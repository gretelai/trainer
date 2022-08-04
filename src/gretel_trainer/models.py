"""Model specific parameters and definitions"""

from collections import namedtuple
from enum import Enum


class BaseConfig:
    """This class should not be used directly, model-specific configs should be
    derived from this class
    """

    def __init__(self, config_file: str, default_rows: int, max_rows: int, default_header_clusters: int, max_header_clusters: int):
        self.config_file = config_file
        self.default_rows = default_rows
        self.max_rows = max_rows
        self.default_header_clusters = default_header_clusters
        self.max_header_clusters = max_header_clusters


class GretelLSTMConfig(BaseConfig):

    config = "synthetics/default"
    default_header_clusters = 20
    max_header_clusters = 30
    default_rows = 50000
    max_rows = 1000000

    def __init__(self):
        super().__init__(
            config_file=self.config,
            default_rows=self.default_rows,
            max_rows=self.max_rows,
            default_header_clusters=self.default_header_clusters,
            max_header_clusters=self.max_header_clusters,
        )


class GretelCTGANConfig(BaseConfig):

    config = "synthetics/high-dimensionality"
    default_header_clusters = 100
    max_header_clusters = 1000
    default_rows = 50000
    max_rows = 1000000

    def __init__(self):
        super().__init__(
            config_file=self.config,
            default_rows=self.default_rows,
            max_rows=self.max_rows,
            default_header_clusters=self.default_header_clusters,
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

    @classmethod
    def get_default_rows(cls, model: str) -> int:
        return cls[model].config.default_rows

    @classmethod
    def get_default_header_clusters(cls, model: str) -> int:
        return cls[model].config.default_header_clusters



class Model(namedtuple("Model", "config"), ExtendedEnum):
    """Enum to pair default models and configurations"""

    GretelLSTM = GretelLSTMConfig()
    GretelCTGAN = GretelCTGANConfig()
