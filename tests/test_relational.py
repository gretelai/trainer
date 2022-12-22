import os
import tempfile

from pathlib import Path

import boto3
import pandas as pd
import pandas.testing as pdtest
import pytest

from botocore import UNSIGNED
from botocore.client import Config
from unittest.mock import Mock, patch

from gretel_trainer.relational.connectors import sqlite_conn
from gretel_trainer.relational.core import MultiTableException, RelationalData
from gretel_trainer.relational.multi_table import (
    GenerateStatus,
    MultiTable,
    TableEvaluation,
    TrainStatus,
)
from gretel_trainer.relational.strategies.ancestral import AncestralStrategy
from gretel_trainer.relational.strategies.single_table import SingleTableStrategy


def _setup_pets():
    humans = pd.DataFrame(
        data={
            "name": ["John", "Paul", "George", "Ringo", "Billy"],
            "city": ["Liverpool", "Liverpool", "Liverpool", "Liverpool", "Houston"],
            "id": [1, 2, 3, 4, 5],
        }
    )
    pets = pd.DataFrame(
        data={
            "name": ["Lennon", "McCartney", "Harrison", "Starr", "Preston"],
            "age": [6, 14, 8, 7, 2],
            "id": [1, 2, 3, 4, 5],
            "human_id": [1, 2, 3, 4, 5],
        }
    )
    rel_data = RelationalData()
    rel_data.add_table("humans", "id", humans)
    rel_data.add_table("pets", "id", pets)
    rel_data.add_foreign_key("pets.human_id", "humans.id")
    return rel_data


def _setup_nba(synthetic: bool = False):
    if synthetic:
        states = ["PA", "FL"]
        cities = ["Philadelphia", "Miami"]
        teams = ["Sixers", "Heat"]
    else:
        states = ["CA", "TN"]
        cities = ["Los Angeles", "Memphis"]
        teams = ["Lakers", "Grizzlies"]

    states = pd.DataFrame(data={"name": states, "id": [1, 2]})
    cities = pd.DataFrame(data={"name": cities, "id": [1, 2], "state_id": [1, 2]})
    teams = pd.DataFrame(data={"name": teams, "id": [1, 2], "city_id": [1, 2]})

    rel_data = RelationalData()
    rel_data.add_table("states", "id", states)
    rel_data.add_table("cities", "id", cities)
    rel_data.add_table("teams", "id", teams)
    rel_data.add_foreign_key("teams.city_id", "cities.id")
    rel_data.add_foreign_key("cities.state_id", "states.id")

    return rel_data, states, cities, teams


def _setup_ecommerce():
    ecommerce = RelationalData()
    ecommerce.add_table(
        "events",
        "id",
        pd.DataFrame(columns=["id", "browser", "traffic_source", "user_id"]),
    )
    ecommerce.add_table(
        "users", "id", pd.DataFrame(columns=["id", "first_name", "last_name"])
    )
    ecommerce.add_table(
        "inventory_items",
        "id",
        pd.DataFrame(
            columns=[
                "id",
                "sold_at",
                "cost",
                "product_id",
                "product_distribution_center_id",
            ]
        ),
    )
    ecommerce.add_table(
        "products",
        "id",
        pd.DataFrame(columns=["id", "name", "brand", "distribution_center_id"]),
    )
    ecommerce.add_table(
        "distribution_center", "id", pd.DataFrame(columns=["id", "name"])
    )
    ecommerce.add_table(
        "order_items",
        "id",
        pd.DataFrame(
            columns=["id", "sale_price", "status", "user_id", "inventory_item_id"]
        ),
    )
    ecommerce.add_foreign_key("events.user_id", "users.id")
    ecommerce.add_foreign_key("order_items.user_id", "users.id")
    ecommerce.add_foreign_key("order_items.inventory_item_id", "inventory_items.id")
    ecommerce.add_foreign_key("inventory_items.product_id", "products.id")
    ecommerce.add_foreign_key(
        "inventory_items.product_distribution_center_id", "distribution_center.id"
    )
    ecommerce.add_foreign_key(
        "products.distribution_center_id", "distribution_center.id"
    )
    return ecommerce


def _setup_mutagenesis():
    mutagenesis = RelationalData()
    mutagenesis.add_table(
        "bond", None, pd.DataFrame(columns=["type", "atom1_id", "atom2_id"])
    )
    mutagenesis.add_table(
        "atom",
        "atom_id",
        pd.DataFrame(columns=["atom_id", "element", "charge", "molecule_id"]),
    )
    mutagenesis.add_table(
        "molecule", "molecule_id", pd.DataFrame(columns=["molecule_id", "mutagenic"])
    )
    mutagenesis.add_foreign_key("bond.atom1_id", "atom.atom_id")
    mutagenesis.add_foreign_key("bond.atom2_id", "atom.atom_id")
    mutagenesis.add_foreign_key("atom.molecule_id", "molecule.molecule_id")
    return mutagenesis


def test_extracting_relational_data():
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    with tempfile.NamedTemporaryFile() as f:
        s3.download_fileobj("gretel-blueprints-pub", "rdb/ecom_xf.db", f)
        sqlite = sqlite_conn(f.name)
        extracted = sqlite.extract()

    all_tables = extracted.list_all_tables()
    assert set(all_tables) == {
        "users",
        "events",
        "products",
        "distribution_center",
        "order_items",
        "inventory_items",
    }

    manual = _setup_ecommerce()

    for table in all_tables:
        assert len(extracted.get_table_data(table)) > 1
        assert extracted.get_parents(table) == manual.get_parents(table)
        assert extracted.get_foreign_keys(table) == manual.get_foreign_keys(table)


def test_extract_subsets_of_relational_data():
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    with tempfile.NamedTemporaryFile() as f:
        s3.download_fileobj("gretel-blueprints-pub", "rdb/ecom_xf.db", f)
        sqlite = sqlite_conn(f.name)

        with pytest.raises(MultiTableException):
            sqlite.extract(only=["users"], ignore=["events"])

        only = sqlite.extract(only=["users", "events", "products"])
        ignore = sqlite.extract(
            ignore=["distribution_center", "order_items", "inventory_items"]
        )

    expected_tables = {"users", "events", "products"}
    assert set(only.list_all_tables()) == expected_tables
    assert set(ignore.list_all_tables()) == expected_tables

    # `products` has a foreign key to `distribution_center` in the source, but because the
    # latter table was not extracted, the relationship is not recognized
    assert only.get_parents("products") == []
    assert ignore.get_parents("products") == []


def test_ecommerce_relational_data():
    ecom = _setup_ecommerce()

    assert ecom.get_parents("users") == []
    assert ecom.get_parents("events") == ["users"]
    assert set(ecom.get_parents("inventory_items")) == {
        "products",
        "distribution_center",
    }

    users_with_ancestors = ecom.get_table_data_with_ancestors("users")
    assert set(users_with_ancestors.columns) == {
        "self|id",
        "self|first_name",
        "self|last_name",
    }
    restored_users = ecom.drop_ancestral_data(users_with_ancestors)
    assert set(restored_users.columns) == {"id", "first_name", "last_name"}

    events_with_ancestors = ecom.get_table_data_with_ancestors("events")
    assert set(events_with_ancestors.columns) == {
        "self|id",
        "self|browser",
        "self|traffic_source",
        "self|user_id",
        "self.user_id|id",
        "self.user_id|first_name",
        "self.user_id|last_name",
    }
    restored_events = ecom.drop_ancestral_data(events_with_ancestors)
    assert set(restored_events.columns) == {
        "id",
        "browser",
        "traffic_source",
        "user_id",
    }

    inventory_items_with_ancestors = ecom.get_table_data_with_ancestors(
        "inventory_items"
    )
    assert set(inventory_items_with_ancestors.columns) == {
        "self|id",
        "self|sold_at",
        "self|cost",
        "self|product_id",
        "self|product_distribution_center_id",
        "self.product_id|id",
        "self.product_id|name",
        "self.product_id|brand",
        "self.product_id|distribution_center_id",
        "self.product_distribution_center_id|id",
        "self.product_distribution_center_id|name",
        "self.product_id.distribution_center_id|id",
        "self.product_id.distribution_center_id|name",
    }
    restored_inventory_items = ecom.drop_ancestral_data(inventory_items_with_ancestors)
    assert set(restored_inventory_items.columns) == {
        "id",
        "sold_at",
        "cost",
        "product_id",
        "product_distribution_center_id",
    }

    # seed dataframes
    assert ecom.build_seed_data_for_table("users", {}) is None

    events_seed = ecom.build_seed_data_for_table(
        "events", {"users": users_with_ancestors}
    )
    assert events_seed is not None and set(events_seed.columns) == {
        "self.user_id|id",
        "self.user_id|first_name",
        "self.user_id|last_name",
    }

    # inventory_items_seed = ecom.build_seed_data_for_table("inventory_items", {
    #     "products": ecom.get_table_data_with_ancestors("products"),
    #     "distribution_center": ecom.get_table_data_with_ancestors("distribution_center"),
    # })
    # assert set(inventory_items_seed.columns) == {
    # # TODO
    # }


def test_mutagenesis_relational_data():
    mutagenesis = _setup_mutagenesis()

    assert mutagenesis.get_parents("bond") == ["atom"]
    assert mutagenesis.get_parents("atom") == ["molecule"]

    assert mutagenesis.get_primary_key("bond") is None
    assert mutagenesis.get_primary_key("atom") == "atom_id"

    bond_with_ancestors = mutagenesis.get_table_data_with_ancestors("bond")
    assert set(bond_with_ancestors.columns) == {
        "self|type",
        "self|atom1_id",
        "self|atom2_id",
        "self.atom1_id|atom_id",
        "self.atom1_id|element",
        "self.atom1_id|charge",
        "self.atom1_id|molecule_id",
        "self.atom2_id|atom_id",
        "self.atom2_id|element",
        "self.atom2_id|charge",
        "self.atom2_id|molecule_id",
        "self.atom1_id.molecule_id|molecule_id",
        "self.atom1_id.molecule_id|mutagenic",
        "self.atom2_id.molecule_id|molecule_id",
        "self.atom2_id.molecule_id|mutagenic",
    }
    restored_bond = mutagenesis.drop_ancestral_data(bond_with_ancestors)
    assert set(restored_bond.columns) == {"type", "atom1_id", "atom2_id"}


def test_ancestral_data_from_different_tablesets():
    rel_data, _, _, _ = _setup_nba()

    # By default, get data from source
    source_teams_with_ancestors = rel_data.get_table_data_with_ancestors("teams")
    assert set(source_teams_with_ancestors["self|name"]) == {"Lakers", "Grizzlies"}

    _, custom_states, custom_cities, custom_teams = _setup_nba(synthetic=True)
    custom_tableset = {
        "states": custom_states,
        "cities": custom_cities,
        "teams": custom_teams,
    }

    # Optionally provide a different tableset
    custom_teams_with_ancestors = rel_data.get_table_data_with_ancestors(
        "teams", custom_tableset
    )
    assert set(custom_teams_with_ancestors["self|name"]) == {"Sixers", "Heat"}


def test_list_ancestral_keys():
    mutagenesis = _setup_mutagenesis()
    assert set(mutagenesis.list_multigenerational_keys("bond")) == {
        "self|atom1_id",
        "self|atom2_id",
        "self.atom1_id|atom_id",
        "self.atom1_id|molecule_id",
        "self.atom2_id|atom_id",
        "self.atom2_id|molecule_id",
        "self.atom1_id.molecule_id|molecule_id",
        "self.atom2_id.molecule_id|molecule_id",
    }


def test_whether_column_is_ancestral():
    mutagenesis = _setup_mutagenesis()
    assert mutagenesis.is_ancestral_column("self|atom1_id") is False
    assert mutagenesis.is_ancestral_column("self.atom1_id|atom1_id")
    assert mutagenesis.is_ancestral_column("self.atom1_id.molecule_id|atom1_id")


def test_relational_data_as_dict():
    ecom = _setup_ecommerce()
    as_dict = ecom.as_dict("test_out")

    assert as_dict["tables"] == {
        "users": {"primary_key": "id", "csv_path": "test_out/users.csv"},
        "events": {"primary_key": "id", "csv_path": "test_out/events.csv"},
        "distribution_center": {
            "primary_key": "id",
            "csv_path": "test_out/distribution_center.csv",
        },
        "products": {"primary_key": "id", "csv_path": "test_out/products.csv"},
        "inventory_items": {
            "primary_key": "id",
            "csv_path": "test_out/inventory_items.csv",
        },
        "order_items": {"primary_key": "id", "csv_path": "test_out/order_items.csv"},
    }
    assert set(as_dict["foreign_keys"]) == {
        ("events.user_id", "users.id"),
        ("order_items.user_id", "users.id"),
        ("order_items.inventory_item_id", "inventory_items.id"),
        ("inventory_items.product_id", "products.id"),
        ("inventory_items.product_distribution_center_id", "distribution_center.id"),
        ("products.distribution_center_id", "distribution_center.id"),
    }


def test_ecommerce_filesystem_serde():
    ecom = _setup_ecommerce()

    with tempfile.TemporaryDirectory() as tmp:
        ecom.to_filesystem(tmp)

        expected_files = [
            f"{tmp}/metadata.json",
            f"{tmp}/events.csv",
            f"{tmp}/users.csv",
            f"{tmp}/distribution_center.csv",
            f"{tmp}/products.csv",
            f"{tmp}/inventory_items.csv",
            f"{tmp}/order_items.csv",
        ]
        for expected_file in expected_files:
            assert os.path.exists(expected_file)

        from_json = RelationalData.from_filesystem(f"{tmp}/metadata.json")

    for table in ecom.list_all_tables():
        pdtest.assert_frame_equal(
            ecom.get_table_data(table), from_json.get_table_data(table)
        )
        assert ecom.get_parents(table) == from_json.get_parents(table)
        assert ecom.get_foreign_keys(table) == from_json.get_foreign_keys(table)


def test_filesystem_serde_accepts_missing_primary_keys():
    mutagenesis = _setup_mutagenesis()

    with tempfile.TemporaryDirectory() as tmp:
        mutagenesis.to_filesystem(tmp)
        from_json = RelationalData.from_filesystem(f"{tmp}/metadata.json")

    assert from_json.get_primary_key("bond") is None
    assert from_json.get_primary_key("atom") == "atom_id"


def test_single_table_strategy_removes_primary_and_foreign_keys_for_training():
    pets = _setup_pets()
    strategy = SingleTableStrategy()

    training_pets = strategy.prepare_training_data("pets", pets)

    assert set(training_pets.columns) == {"name", "age"}


def test_single_table_strategy_retrains_same_tables_only():
    ecom = _setup_ecommerce()
    strategy = SingleTableStrategy()
    assert set(strategy.tables_to_retrain(["users"], ecom)) == {"users"}
    assert set(strategy.tables_to_retrain(["users", "events"], ecom)) == {
        "users",
        "events",
    }
    assert set(strategy.tables_to_retrain(["products"], ecom)) == {"products"}


def test_ancestral_strategy_prepares_multigenerational_data_without_keys_or_highly_unique_categorial_fields():
    pets = _setup_pets()
    strategy = AncestralStrategy()

    training_pets = strategy.prepare_training_data("pets", pets)

    assert set(training_pets.columns) == {
        "self|name",
        "self|age",
        "self.human_id|city",
        # self|id rejected (primary key)
        # self|human_id rejected (foreign key)
        # self.human_id|id rejected (primary key)
        # self.human_id|name rejected (highly unique categorical)
    }


def test_ancestral_strategy_retrains_tables_and_their_children():
    ecom = _setup_ecommerce()
    strategy = AncestralStrategy()
    assert set(strategy.tables_to_retrain(["users"], ecom)) == {
        "users",
        "events",
        "order_items",
    }
    assert set(strategy.tables_to_retrain(["products"], ecom)) == {
        "products",
        "inventory_items",
        "order_items",
    }
    assert set(strategy.tables_to_retrain(["users", "products"], ecom)) == {
        "users",
        "events",
        "products",
        "inventory_items",
        "order_items",
    }


def test_training_through_trainer():
    pets = _setup_pets()

    with tempfile.TemporaryDirectory() as work_dir, patch(
        "gretel_trainer.trainer.create_or_get_unique_project"
    ), patch("gretel_trainer.trainer.Trainer.train") as train, patch(
        "gretel_trainer.trainer.Trainer.trained_successfully"
    ) as trained_successfully:
        trained_successfully.return_value = True
        multitable = MultiTable(pets, working_dir=work_dir)

        # Need to patch configure_session in two spots because MultiTable calls it first
        # (before any parallelization) and then each Trainer instance calls it internally
        with patch("gretel_trainer.relational.multi_table.configure_session"), patch(
            "gretel_trainer.trainer.configure_session"
        ):
            multitable.train()

        for table in ["humans", "pets"]:
            training_csv = Path(f"{work_dir}/{table}-train.csv")
            assert os.path.exists(training_csv)
            train.assert_any_call(training_csv)


def test_evaluate():
    rel_data, _, _, _ = _setup_nba()
    _, syn_states, syn_cities, syn_teams = _setup_nba(synthetic=True)

    multitable = MultiTable(rel_data)

    with patch(
        "gretel_trainer.relational.multi_table.Trainer.load"
    ) as load_trainer, patch(
        "gretel_trainer.relational.multi_table.QualityReport"
    ) as quality_report:
        trainer = Mock()
        trainer.get_sqs_score.return_value = 42
        load_trainer.return_value = trainer

        report = quality_report.return_value
        report.run.return_value = None
        report.peek = lambda: {"score": 84}

        multitable.train_statuses["cities"] = TrainStatus.Completed

        evaluations = multitable.evaluate(
            {
                "states": syn_states,
                "cities": syn_cities,
                "teams": syn_teams,
            }
        )

    assert evaluations["states"] == TableEvaluation(individual_sqs=84, ancestral_sqs=84)
    assert evaluations["cities"] == TableEvaluation(individual_sqs=42, ancestral_sqs=84)


def test_single_table_generation_readiness():
    ecom = _setup_ecommerce()
    strategy = SingleTableStrategy()

    # All tables are immediately ready for generation
    assert set(strategy.ready_to_generate(ecom, [], [])) == {
        "users",
        "events",
        "distribution_center",
        "products",
        "inventory_items",
        "order_items",
    }

    # Tables that are in progress or finished are no longer ready
    assert set(strategy.ready_to_generate(ecom, ["users"], ["events"])) == {
        "distribution_center",
        "products",
        "inventory_items",
        "order_items",
    }


def test_ancestral_generation_readiness():
    ecom = _setup_ecommerce()
    strategy = AncestralStrategy()

    # To start, "eldest generation" tables (those with no parents / outbound foreign keys) are ready
    assert set(strategy.ready_to_generate(ecom, [], [])) == {
        "users",
        "distribution_center",
    }

    # Once a table has been started, it is no longer ready
    assert set(strategy.ready_to_generate(ecom, ["users"], [])) == {
        "distribution_center"
    }

    # It's possible to be in a state where work is happening but nothing is ready
    assert (
        set(strategy.ready_to_generate(ecom, ["users", "distribution_center"], []))
        == set()
    )

    # `events` was only blocked by `users`; once the latter completes, the former is ready,
    # regardless of the state of the unrelated `distribution_center` table
    assert set(
        strategy.ready_to_generate(ecom, ["distribution_center"], ["users"])
    ) == {"events"}

    # Similarly, the completion of `distribution_center` unblocks `products`,
    # regardless of progress on `events`
    assert set(
        strategy.ready_to_generate(ecom, [], ["users", "distribution_center"])
    ) == {"events", "products"}

    # Remaining tables become ready as their parents complete
    assert set(
        strategy.ready_to_generate(
            ecom, [], ["users", "distribution_center", "events", "products"]
        )
    ) == {"inventory_items"}

    # As above, being in progress is not enough! Work is happening but nothing new is ready
    assert (
        set(
            strategy.ready_to_generate(
                ecom,
                ["inventory_items"],
                ["users", "distribution_center", "events", "products"],
            )
        )
        == set()
    )

    assert set(
        strategy.ready_to_generate(
            ecom,
            [],
            ["users", "distribution_center", "events", "products", "inventory_items"],
        )
    ) == {"order_items"}

    assert (
        set(
            strategy.ready_to_generate(
                ecom,
                ["order_items"],
                [
                    "users",
                    "distribution_center",
                    "events",
                    "products",
                    "inventory_items",
                ],
            )
        )
        == set()
    )
    assert (
        set(
            strategy.ready_to_generate(
                ecom,
                [],
                [
                    "users",
                    "distribution_center",
                    "events",
                    "products",
                    "inventory_items",
                    "order_items",
                ],
            )
        )
        == set()
    )
