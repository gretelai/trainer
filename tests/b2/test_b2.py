from pathlib import Path

from gretel_trainer.b2 import compare


class DoNothingModel:
    def train(self, source, **kwargs):
        pass

    def generate(self, **kwargs):
        return Path()


def test_things_work(iris):
    comparison = compare(
        datasets=[iris],
        models=[DoNothingModel],
    ).wait()

    assert len(comparison.results) == 1


def test_run_with_gretel_dataset():
    pass


def test_run_with_custom_csv_dataset():
    pass


def test_run_with_custom_dataframe_dataset():
    pass


# parametrize
def test_run_with_default_gretel_model_through_sdk():
    pass


def test_run_with_specific_gretel_model_config():
    pass


def test_run_with_custom_model():
    pass


def test_gptx_skips_too_many_columns():
    pass


def test_gptx_skips_non_natural_language_datatype():
    pass


def test_lstm_skips_datasets_with_over_150_columns():
    pass