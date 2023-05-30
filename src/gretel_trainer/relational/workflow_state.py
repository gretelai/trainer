from dataclasses import dataclass, field

from gretel_client.projects.models import Model
from gretel_client.projects.records import RecordHandler


@dataclass
class Classify:
    models: dict[str, Model] = field(default_factory=dict)


@dataclass
class TransformsTrain:
    models: dict[str, Model] = field(default_factory=dict)
    lost_contact: list[str] = field(default_factory=list)


@dataclass
class SyntheticsTrain:
    models: dict[str, Model] = field(default_factory=dict)
    lost_contact: list[str] = field(default_factory=list)


@dataclass
class SyntheticsRun:
    identifier: str
    record_size_ratio: float
    preserved: list[str]
    record_handlers: dict[str, RecordHandler]
    lost_contact: list[str]
