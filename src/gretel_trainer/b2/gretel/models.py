from pathlib import Path
from typing import Dict, Optional, Union

from gretel_client.projects.exceptions import ModelConfigError
from gretel_client.projects.models import read_model_config

import gretel_trainer
from gretel_trainer import models
from gretel_trainer.b2.core import BenchmarkException, Dataset, Datatype

GretelModelConfig = Union[str, Path, Dict]


TRAINER_MODEL_TYPE_CONSTRUCTORS = {
    "actgan": models.GretelACTGAN,
    "amplify": models.GretelAmplify,
    "synthetics": models.GretelLSTM,
    # Benchmark GretelAuto sends None model_type to Trainer to trigger dynamic model selection
    "AUTO": lambda _: None,
}


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
    def trainer_model_type(self) -> Optional[gretel_trainer.models._BaseConfig]:
        constructor = TRAINER_MODEL_TYPE_CONSTRUCTORS.get(self.model_key)
        if constructor is None:
            return None
        else:
            return constructor(config=self.config)

    def runnable(self, dataset: Dataset) -> bool:
        if self.model_key == "synthetics":
            return dataset.column_count <= 150
        elif self.model_key == "gpt_x":
            return (
                dataset.column_count == 1
                and dataset.datatype == Datatype.natural_language
            )
        else:
            return True


# Defaults


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


class GretelDGAN(GretelModel):
    config = "synthetics/time-series"
