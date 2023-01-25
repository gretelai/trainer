import tempfile
from unittest.mock import patch

import pytest

from gretel_trainer.relational.core import MultiTableException
from gretel_trainer.relational.multi_table import MultiTable
from gretel_trainer.relational.strategies.ancestral import AncestralStrategy
from gretel_trainer.relational.strategies.independent import IndependentStrategy


def test_model_strategy_combinations(ecom):
    with tempfile.TemporaryDirectory() as tmpdir, patch(
        "gretel_trainer.relational.multi_table.configure_session"
    ), patch("gretel_trainer.relational.multi_table.create_or_get_unique_project"):

        # Default to Amplify/single-table
        mt = MultiTable(ecom, working_dir=tmpdir)
        assert mt._model_config == "synthetics/amplify"
        assert isinstance(mt._strategy, IndependentStrategy)

        # Default to Amplify when ancestral strategy is chosen
        mt = MultiTable(ecom, working_dir=tmpdir, strategy="ancestral")
        assert mt._model_config == "synthetics/amplify"
        assert isinstance(mt._strategy, AncestralStrategy)

        # Cross-table only works with Amplify
        with pytest.raises(MultiTableException):
            MultiTable(
                ecom,
                working_dir=tmpdir,
                strategy="ancestral",
                gretel_model="actgan",
            )
        with pytest.raises(MultiTableException):
            MultiTable(
                ecom,
                working_dir=tmpdir,
                strategy="ancestral",
                gretel_model="lstm",
            )

        # Independent strategy works with Amplify, ACTGAN, and LSTM
        MultiTable(
            ecom,
            working_dir=tmpdir,
            strategy="independent",
            gretel_model="amplify",
        )
        MultiTable(
            ecom,
            working_dir=tmpdir,
            strategy="independent",
            gretel_model="actgan",
        )
        MultiTable(
            ecom,
            working_dir=tmpdir,
            strategy="independent",
            gretel_model="lstm",
        )


def test_refresh_interval_config(ecom):
    with tempfile.TemporaryDirectory() as tmpdir, patch(
        "gretel_trainer.relational.multi_table.configure_session"
    ), patch("gretel_trainer.relational.multi_table.create_or_get_unique_project"):
        # default to 180
        mt = MultiTable(ecom, working_dir=tmpdir)
        assert mt._refresh_interval == 180

        # must be at least 60
        mt = MultiTable(ecom, working_dir=tmpdir, refresh_interval=30)
        assert mt._refresh_interval == 60

        # may be greater than 60
        mt = MultiTable(ecom, working_dir=tmpdir, refresh_interval=120)
        assert mt._refresh_interval == 120
