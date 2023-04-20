import copy
from inspect import isclass
from pathlib import Path
from typing import Dict, Optional, Type, Union

from gretel_client.projects.exceptions import ModelConfigError
from gretel_client.projects.models import read_model_config

import gretel_trainer
from gretel_trainer import models
from gretel_trainer.benchmark.core import BenchmarkException, Dataset, Datatype

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


class _GretelModelWithOverrides(GretelModel):

    _delegate: GretelModel
    _name: str
    _extra_config: Optional[dict]

    def __init__(self,
                 delegate: GretelModel,
                 name: str = "",
                 extra_config: Optional[dict] = None,
    ):
        self._delegate = delegate
        self._name = name
        self._extra_config = extra_config

    @property
    def name(self) -> str:
        return self._name if self._name else self._delegate.name

    @property
    def config(self) -> GretelModelConfig:
        if not self._extra_config:
            return self._delegate.config

        cfg = copy.deepcopy(read_model_config(self._delegate.config))
        model_cfg = next(c for c in cfg["models"][0].values())
        _recursive_dict_update(model_cfg, self._extra_config)
        return cfg

    @property
    def model_key(self) -> str:
        return self._delegate.model_key

    @property
    def trainer_model_type(self) -> Optional[gretel_trainer.models._BaseConfig]:
        return self._delegate.trainer_model_type

    def runnable(self, dataset: Dataset) -> bool:
        return self._delegate.runnable(dataset)


def configure_model(
        model: Union[Type[GretelModel], GretelModel],
        name: str = "",
        extra_config: Optional[dict] = None,
) -> GretelModel:
    """Returns a GretelModel with an updated model configuration and/or name.

    This function works both on a GretelModel subclass as well as on a GretelModel
    instance. The result is always a GretelModel instance.

    The returned GretelModel differs from the input in only two aspects:
    - the name is possibly changed, and
    - the model config has been overridden according to `config_override`.

    Args:
        model:
            Model class or instance to modify.
        name:
            A custom name for the model.
        config_override:
            A dict that is applied as an override to the model config.

    Returns:
        A GretelModel instance..
    """
    if isclass(model):
        model = model()

    return _GretelModelWithOverrides(delegate=model, name=name, extra_config=extra_config)


def _recursive_dict_update(a: dict, b: dict, _ctx: str = "") -> dict:
    for k, vb in b.items():
        if not isinstance(va := a.get(k, None), dict):
            # Overwrite non-dict entries
            a[k] = vb
        elif not isinstance(vb, dict):
            raise ValueError(f"cannot merge non-dict value into dict at key {_ctx}{k}")
        else:
            _recursive_dict_update(va, vb, _ctx=f"{_ctx}{k}.")
    return a
