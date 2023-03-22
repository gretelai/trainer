from typing import Optional

from gretel_trainer.b2.core import BenchmarkConfig, Dataset, run_out_path, RunIdentifier, Timer
from gretel_trainer.b2.custom.models import CustomModel


class CustomStrategy:
    def __init__(
        self,
        model: CustomModel,
        dataset: Dataset,
        run_identifier: RunIdentifier,
        config: BenchmarkConfig,
    ):
        self.model = model
        self.dataset = dataset
        self.run_identifier = run_identifier
        self.working_dir = config.working_dir
        self.train_timer: Optional[Timer] = None
        self.generate_timer: Optional[Timer] = None

    def train(self) -> None:
        self.train_timer = Timer()
        with self.train_timer:
            self.model.train(self.dataset)

    def generate(self) -> None:
        synthetic_data_path = run_out_path(self.working_dir, self.run_identifier)
        self.generate_timer = Timer()
        with self.generate_timer:
            synthetic_df = self.model.generate(
                num_records=self.dataset.row_count,
            )
            synthetic_df.to_csv(synthetic_data_path, index=False)

    def get_sqs_score(self) -> int:
        # TODO
        return 0

    def runnable(self, dataset: Dataset) -> bool:
        return True

    def get_train_time(self) -> Optional[float]:
        return _get_duration(self.train_timer)

    def get_generate_time(self) -> Optional[float]:
        return _get_duration(self.generate_timer)


def _get_duration(timer: Optional[Timer]) -> Optional[float]:
    if timer is None:
        return None
    else:
        return timer.duration()
