from importlib.metadata import PackageNotFoundError, version


def get_trainer_version() -> str:
    try:
        return version("gretel_trainer")
    except PackageNotFoundError:
        return "undefined"
