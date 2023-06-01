import pytest

from gretel_trainer.relational.subset import SubsetConfig


def test_subset_config():
    # Can't have a target row count <= 0
    with pytest.raises(ValueError):
        SubsetConfig(target_row_count=0)

    # Concrete row count
    config = SubsetConfig(target_row_count=100)
    assert config.calculate_row_count(100) == 100

    # Ratio
    config = SubsetConfig(target_row_count=0.5)
    assert config.calculate_row_count(100) == 50
