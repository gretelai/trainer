from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import gretel_trainer

from gretel_client.projects.exceptions import ModelConfigError
from gretel_client.projects.models import read_model_config
from gretel_trainer import models
from gretel_trainer.benchmark.core import BenchmarkException, DataSource, Datatype

GretelModelConfig = Union[str, Path, Dict]


TRAINER_MODEL_TYPE_CONSTRUCTORS = {
    "actgan": models.GretelACTGAN,
    "amplify": models.GretelAmplify,
    "synthetics": models.GretelLSTM,

    # Benchmark GretelAuto sends None model_type to Trainer to trigger dynamic model selection
    "AUTO": lambda _: None,
}


def _is_compatible(model_key: str, source: DataSource) -> bool:
    if model_key == "synthetics":
        return source.column_count <= 150
    elif model_key == "gpt_x":
        return source.column_count == 1 and source.datatype == Datatype.NATURAL_LANGUAGE
    else:
        return True


class GretelModel:
    config: GretelModelConfig

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def model_key(self) -> str:
        if self.config == "AUTO":
            return "AUTO"

        try:
            config_dict = read_model_config(self.config)
            return list(config_dict["models"][0])[0]
        except (ModelConfigError, KeyError):
            raise BenchmarkException(f"Invalid Gretel model config")

    @property
    def use_trainer(self) -> bool:
        return self.model_key in TRAINER_MODEL_TYPE_CONSTRUCTORS.keys()

    @property
    def trainer_model_type(self) -> Optional[gretel_trainer.models._BaseConfig]:
        return TRAINER_MODEL_TYPE_CONSTRUCTORS[self.model_key](self.config)

    def runnable(self, source: DataSource) -> bool:
        return _is_compatible(self.model_key, source)


class GretelLSTM(GretelModel):
    config = "synthetics/tabular-lstm"

class GretelAmplify(GretelModel):
    config = "synthetics/amplify"

class GretelAuto(GretelModel):
    config = "AUTO"

class GretelACTGAN(GretelModel):
    config = "synthetics/tabular-actgan"

class GretelGPTX(GretelModel):
    config = "synthetics/natural-language"
