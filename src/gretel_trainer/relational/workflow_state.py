from dataclasses import dataclass, field
from typing import Dict, List

from gretel_client.projects.models import Model
from gretel_client.projects.records import RecordHandler


@dataclass
class TransformsTrain:
    models: Dict[str, Model] = field(default_factory=dict)
    lost_contact: List[str] = field(default_factory=list)


@dataclass
class SyntheticsTrain:
    models: Dict[str, Model] = field(default_factory=dict)
    lost_contact: List[str] = field(default_factory=list)
    training_columns: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class SyntheticsRun:
    identifier: str
    record_size_ratio: float
    preserved: List[str]
    record_handlers: Dict[str, RecordHandler]
    lost_contact: List[str]
    missing_model: List[str]
