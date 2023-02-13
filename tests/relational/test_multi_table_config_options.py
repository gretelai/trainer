import tempfile
from unittest.mock import patch

import pytest

from gretel_trainer.relational.core import MultiTableException
from gretel_trainer.relational.multi_table import MultiTable
from gretel_trainer.relational.strategies.ancestral import AncestralStrategy
from gretel_trainer.relational.strategies.independent import IndependentStrategy


def test_model_strategy_combinations(ecom, project):
    with tempfile.TemporaryDirectory() as tmpdir, patch(
        "gretel_trainer.relational.multi_table._upload_gretel_backup"
    ):
        project.name = tmpdir
        project.upload_artifact.return_value = "gretel_abcdefg_somefile.someextension"

        # Default to Amplify/single-table
        mt = MultiTable(ecom, project_display_name=tmpdir)
        assert mt._model_config == "synthetics/amplify"
        assert isinstance(mt._strategy, IndependentStrategy)

        # Default to Amplify when ancestral strategy is chosen
        mt = MultiTable(ecom, project_display_name=tmpdir, strategy="ancestral")
        assert mt._model_config == "synthetics/amplify"
        assert isinstance(mt._strategy, AncestralStrategy)

        # Cross-table only works with Amplify
        with pytest.raises(MultiTableException):
            MultiTable(
                ecom,
                project_display_name=tmpdir,
                strategy="ancestral",
                gretel_model="actgan",
            )
        with pytest.raises(MultiTableException):
            MultiTable(
                ecom,
                project_display_name=tmpdir,
                strategy="ancestral",
                gretel_model="lstm",
            )

        # Independent strategy works with Amplify, ACTGAN, and LSTM
        MultiTable(
            ecom,
            project_display_name=tmpdir,
            strategy="independent",
            gretel_model="amplify",
        )
        MultiTable(
            ecom,
            project_display_name=tmpdir,
            strategy="independent",
            gretel_model="actgan",
        )
        MultiTable(
            ecom,
            project_display_name=tmpdir,
            strategy="independent",
            gretel_model="lstm",
        )


def test_refresh_interval_config(ecom, project):
    with tempfile.TemporaryDirectory() as tmpdir, patch(
        "gretel_trainer.relational.multi_table._upload_gretel_backup"
    ):
        project.name = tmpdir
        project.upload_artifact.return_value = "gretel_abcdefg_somefile.someextension"

        # default to 60
        mt = MultiTable(ecom, project_display_name=tmpdir)
        assert mt._refresh_interval == 60

        # must be at least 30
        mt = MultiTable(ecom, project_display_name=tmpdir, refresh_interval=20)
        assert mt._refresh_interval == 30

        # may be greater than 30
        mt = MultiTable(ecom, project_display_name=tmpdir, refresh_interval=45)
        assert mt._refresh_interval == 45
