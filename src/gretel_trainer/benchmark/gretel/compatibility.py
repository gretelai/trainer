from typing import Optional

from gretel_trainer.benchmark.core import DataSource, Datatype


def is_runnable(model_key: Optional[str], source: DataSource) -> bool:
    if model_key is None:
        return True
    elif model_key in ("lstm", "synthetics"):
        return _lstm(source)
    elif model_key in ("ctgan", "actgan"):
        return _ctgan(source)
    elif model_key in ("gpt_x"):
        return _gptx(source)
    elif model_key in ("amplify"):
        return _amplify(source)
    else:
        return True


def _lstm(source: DataSource) -> bool:
    return source.column_count <= 150


def _ctgan(source: DataSource) -> bool:
    return True


def _gptx(source: DataSource) -> bool:
    return source.column_count == 1 and source.datatype == Datatype.NATURAL_LANGUAGE


def _amplify(source: DataSource) -> bool:
    return True
