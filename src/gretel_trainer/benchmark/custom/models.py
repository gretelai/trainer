import pandas as pd

from gretel_trainer.benchmark.core import DataSource, Evaluator, Model


class CustomExecutor:
    def __init__(self, model: Model, evaluate: Evaluator):
        self.model = model
        self.evaluate = evaluate

    @property
    def model_name(self) -> str:
        return type(self.model).__name__

    def runnable(self, source: DataSource) -> bool:
        return True

    def train(self, source: str, **kwargs) -> None:
        self.model.train(source, **kwargs)

    def generate(self, **kwargs) -> pd.DataFrame:
        return self.model.generate(**kwargs)

    def get_sqs_score(self, synthetic: pd.DataFrame, reference: str) -> int:
        return self.evaluate(synthetic=synthetic, reference=reference)
