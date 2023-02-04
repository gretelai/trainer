from pathlib import Path
from typing import Dict, Union


GretelModelConfig = Union[str, Path, Dict]


class GretelModel:
    config: GretelModelConfig

    @property
    def name(self) -> str:
        return type(self).__name__


# Defaults

class GretelLSTM(GretelModel):
    config = "synthetics/tabular-lstm"


class GretelAmplify(GretelModel):
    config = "synthetics/amplify"


# class GretelAuto(GretelModel):
#     config = "AUTO"


class GretelACTGAN(GretelModel):
    config = "synthetics/tabular-actgan"


class GretelGPTX(GretelModel):
    config = "synthetics/natural-language"


class GretelDGAN(GretelModel):
    config = "synthetics/time-series"
