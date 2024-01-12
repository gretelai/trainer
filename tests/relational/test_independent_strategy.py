import os
import tempfile

from collections import defaultdict
from unittest.mock import patch

import pandas as pd
import pandas.testing as pdtest

from gretel_trainer.relational.core import RelationalData
from gretel_trainer.relational.strategies.independent import IndependentStrategy


def test_preparing_training_data_does_not_mutate_source_data(pets):
    original_tables = {
        table: pets.get_table_data(table).copy() for table in pets.list_all_tables()
    }

    strategy = IndependentStrategy()

    with tempfile.NamedTemporaryFile() as pets_dest, tempfile.NamedTemporaryFile() as humans_dest:
        strategy.prepare_training_data(
            pets, {"pets": pets_dest.name, "humans": humans_dest.name}
        )

    for table in pets.list_all_tables():
        pdtest.assert_frame_equal(original_tables[table], pets.get_table_data(table))


def test_prepare_training_data_removes_primary_and_foreign_keys(pets):
    strategy = IndependentStrategy()

    with tempfile.NamedTemporaryFile() as pets_dest, tempfile.NamedTemporaryFile() as humans_dest:
        training_data = strategy.prepare_training_data(
            pets, {"pets": pets_dest.name, "humans": humans_dest.name}
        )
        train_pets = pd.read_csv(training_data["pets"])

    assert set(train_pets.columns) == {"name", "age"}


def test_prepare_training_data_subset_of_tables(pets):
    strategy = IndependentStrategy()

    with tempfile.NamedTemporaryFile() as pets_dest, tempfile.NamedTemporaryFile() as humans_dest:
        training_data = strategy.prepare_training_data(
            pets, {"humans": humans_dest.name}
        )
        assert not pd.read_csv(training_data["humans"]).empty
        assert os.stat(pets_dest.name).st_size == 0


def test_prepare_training_data_join_table(insurance):
    strategy = IndependentStrategy()

    with tempfile.NamedTemporaryFile() as beneficiary_dest, tempfile.NamedTemporaryFile() as policies_dest:
        training_data = strategy.prepare_training_data(
            insurance,
            {
                "beneficiary": beneficiary_dest.name,
                "insurance_policies": policies_dest.name,
            },
        )
        assert set(training_data.keys()) == {"beneficiary"}
        assert not pd.read_csv(training_data["beneficiary"]).empty
        assert os.stat(policies_dest.name).st_size == 0


def test_retraining_a_set_of_tables_only_retrains_those_tables(ecom):
    strategy = IndependentStrategy()
    assert set(strategy.tables_to_retrain(["users"], ecom)) == {"users"}
    assert set(strategy.tables_to_retrain(["users", "events"], ecom)) == {
        "users",
        "events",
    }
    assert set(strategy.tables_to_retrain(["products"], ecom)) == {"products"}


def test_table_generation_readiness(ecom):
    strategy = IndependentStrategy()

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


def test_generation_job_requests_num_records(pets, output_handler):
    strategy = IndependentStrategy()
    job = strategy.get_generation_job("pets", pets, 2.0, {}, "run-id", output_handler)

    assert job == {"params": {"num_records": 10}}


def test_post_processing_one_to_one(pets):
    strategy = IndependentStrategy()

    # Models train on data with PKs and FKs removed,
    # so those fields won't be present in raw results
    raw_synth_tables = {
        "humans": pd.DataFrame(
            data={
                "name": ["Michael", "Dominique", "Dirk"],
                "city": ["Chicago", "Atlanta", "Dallas"],
            }
        ),
        "pets": pd.DataFrame(
            data={
                "name": ["Bull", "Hawk", "Maverick"],
                "age": [6, 0, 1],
            }
        ),
    }

    # Normally we shuffle synthesized keys for realism, but for deterministic testing we sort instead
    with patch("random.shuffle", wraps=sorted):
        processed = strategy.post_process_synthetic_results(
            raw_synth_tables, [], pets, 1
        )

    # Fields from the raw results do not change
    pdtest.assert_frame_equal(
        processed["humans"],
        pd.DataFrame(
            data={
                "name": ["Michael", "Dominique", "Dirk"],
                "city": ["Chicago", "Atlanta", "Dallas"],
                "id": [0, 1, 2],
            }
        ),
    )
    pdtest.assert_frame_equal(
        processed["pets"],
        pd.DataFrame(
            data={
                "name": ["Bull", "Hawk", "Maverick"],
                "age": [6, 0, 1],
                "id": [0, 1, 2],
                "human_id": [0, 1, 2],
            }
        ),
    )


def test_post_processing_foreign_keys_with_skewed_frequencies_and_different_size_tables(
    trips,
):
    strategy = IndependentStrategy()

    # Simulate a record_size_ratio of 1.5
    raw_synth_tables = {
        "vehicle_types": pd.DataFrame(
            data={"name": ["car", "train", "plane", "bus", "walk", "bike"]}
        ),
        "trips": pd.DataFrame(data={"purpose": ["w"] * 150}),
    }

    processed = strategy.post_process_synthetic_results(
        raw_synth_tables, [], trips, 1.5
    )
    processed_trips = processed["trips"]

    fk_values = set(processed["trips"]["vehicle_type_id"])
    assert fk_values == {0, 1, 2, 3, 4, 5}

    fk_value_counts = defaultdict(int)
    for _, row in processed_trips.iterrows():
        fk_value = row["vehicle_type_id"]
        fk_value_counts[fk_value] = fk_value_counts[fk_value] + 1

    fk_value_counts = sorted(list(fk_value_counts.values()))

    assert fk_value_counts == [5, 5, 15, 30, 35, 60]


# In this scenario, a table (shipping_notifications) has a FK (customer_id) pointing to
# a column that is itself a FK but *not* a PK (orders.customer_id).
# (No, this is not a "perfectly normalized" schema, but it can happen in the wild.)
# We need to ensure tables have FKs synthesized in parent->child order to avoid blowing up
# due to missing columns.
def test_post_processing_fks_to_non_pks(tmpdir):
    rel_data = RelationalData(directory=tmpdir)

    rel_data.add_table(
        name="customers",
        primary_key="id",
        data=pd.DataFrame(data={"id": [1, 2], "name": ["Xavier", "Yesenia"]}),
    )
    rel_data.add_table(
        name="orders",
        primary_key="id",
        data=pd.DataFrame(
            data={
                "id": [1, 2],
                "customer_id": [1, 2],
                "total": [42, 43],
            }
        ),
    )
    rel_data.add_table(
        name="shipping_notifications",
        primary_key="id",
        data=pd.DataFrame(
            data={
                "id": [1, 2],
                "order_id": [1, 2],
                "customer_id": [1, 2],
                "service": ["FedEx", "USPS"],
            }
        ),
    )

    # Add FKs. The third one is the critical one for this test.
    rel_data.add_foreign_key_constraint(
        table="orders",
        constrained_columns=["customer_id"],
        referred_table="customers",
        referred_columns=["id"],
    )
    rel_data.add_foreign_key_constraint(
        table="shipping_notifications",
        constrained_columns=["order_id"],
        referred_table="orders",
        referred_columns=["id"],
    )
    rel_data.add_foreign_key_constraint(
        table="shipping_notifications",
        constrained_columns=["customer_id"],
        referred_table="orders",
        referred_columns=["customer_id"],
    )

    strategy = IndependentStrategy()

    # This dict is deliberately ordered child->parent for this unit test.
    # Were it not for logic in the strategy (processing tables in parent->child order),
    # this setup would cause an exception.
    raw_synth_tables = {
        "shipping_notifications": pd.DataFrame(data={"service": ["FedEx", "USPS"]}),
        "orders": pd.DataFrame(data={"total": [55, 56]}),
        "customers": pd.DataFrame(data={"name": ["Alice", "Bob"]}),
    }

    processed = strategy.post_process_synthetic_results(
        raw_synth_tables, [], rel_data, 1
    )

    for table in rel_data.list_all_tables():
        assert set(processed[table].columns) == set(rel_data.get_table_columns(table))


def test_post_processing_null_foreign_key(tmpdir):
    rel_data = RelationalData(directory=tmpdir)

    rel_data.add_table(
        name="customers",
        primary_key="id",
        data=pd.DataFrame(data={"id": [1, 2], "name": ["Xavier", "Yesenia"]}),
    )
    rel_data.add_table(
        name="events",
        primary_key="id",
        data=pd.DataFrame(
            data={
                "id": [1, 2],
                "customer_id": [1, None],
                "total": [42, 43],
            }
        ),
    )
    rel_data.add_foreign_key_constraint(
        table="events",
        constrained_columns=["customer_id"],
        referred_table="customers",
        referred_columns=["id"],
    )

    strategy = IndependentStrategy()

    raw_synth_tables = {
        "events": pd.DataFrame(data={"total": [55, 56, 57, 58]}),
        "customers": pd.DataFrame(
            data={"name": ["Alice", "Bob", "Christina", "David"]}
        ),
    }

    # Patch shuffle for deterministic testing, but don't swap in `sorted`
    # because that function doesn't cooperate with `None` (raises TypeError)
    with patch("random.shuffle", wraps=lambda x: x):
        processed = strategy.post_process_synthetic_results(
            raw_synth_tables, [], rel_data, 2
        )

    # Given 50% of source FKs are null and record_size_ratio=2,
    # we expect 2/4 customer_ids to be null
    pdtest.assert_frame_equal(
        processed["events"],
        pd.DataFrame(
            data={
                "total": [55, 56, 57, 58],
                "id": [0, 1, 2, 3],
                "customer_id": [None, None, 0, 1],
            }
        ),
    )


def test_post_processing_null_composite_foreign_key(tmpdir):
    rel_data = RelationalData(directory=tmpdir)

    rel_data.add_table(
        name="customers",
        primary_key="id",
        data=pd.DataFrame(
            data={
                "id": [1, 2],
                "first": ["Albert", "Betsy"],
                "last": ["Anderson", "Bond"],
            }
        ),
    )
    rel_data.add_table(
        name="events",
        primary_key="id",
        data=pd.DataFrame(
            data={
                "id": [1, 2, 3, 4, 5],
                "customer_first": ["Albert", "Betsy", None, "Betsy", None],
                "customer_last": ["Anderson", "Bond", None, None, "Bond"],
                "total": [42, 43, 44, 45, 46],
            }
        ),
    )
    rel_data.add_foreign_key_constraint(
        table="events",
        constrained_columns=["customer_first", "customer_last"],
        referred_table="customers",
        referred_columns=["first", "last"],
    )

    strategy = IndependentStrategy()

    raw_synth_tables = {
        "events": pd.DataFrame(data={"total": [55, 56, 57, 58, 59]}),
        "customers": pd.DataFrame(
            data={
                "first": ["Herbert", "Isabella", "Jack", "Kevin", "Louise"],
                "last": ["Hoover", "Irvin", "Johnson", "Knight", "Lane"],
            }
        ),
    }

    # Patch shuffle for deterministic testing
    with patch("random.shuffle", wraps=sorted):
        processed = strategy.post_process_synthetic_results(
            raw_synth_tables, [], rel_data, 1
        )

    # We do not create composite foreign key values with nulls,
    # even if some existed in the source data.
    pdtest.assert_frame_equal(
        processed["events"],
        pd.DataFrame(
            data={
                "total": [55, 56, 57, 58, 59],
                "id": [0, 1, 2, 3, 4],
                "customer_first": ["Herbert", "Isabella", "Jack", "Kevin", "Louise"],
                "customer_last": ["Hoover", "Irvin", "Johnson", "Knight", "Lane"],
            }
        ),
    )


def test_post_processing_with_bypass_table(insurance):
    strategy = IndependentStrategy()

    raw_synth_tables = {
        "beneficiary": pd.DataFrame(
            data={
                "name": ["Adam", "Beth", "Chris", "Demi", "Eric"],
            }
        ),
        "insurance_policies": pd.DataFrame(index=range(40)),
    }

    processed = strategy.post_process_synthetic_results(
        raw_synth_tables, [], insurance, 1
    )

    beneficiary_ids = [0, 1, 2, 3, 4]
    pdtest.assert_frame_equal(
        processed["beneficiary"],
        pd.DataFrame(
            data={
                "name": ["Adam", "Beth", "Chris", "Demi", "Eric"],
                "id": beneficiary_ids,
            }
        ),
    )
    assert set(processed["insurance_policies"].columns) == {
        "id",
        "primary_beneficiary",
        "secondary_beneficiary",
    }
    assert list(processed["insurance_policies"]["id"].values) == list(range(40))
    assert all(
        [
            v in beneficiary_ids
            for v in processed["insurance_policies"]["primary_beneficiary"].values
        ]
    )
    assert all(
        [
            v in beneficiary_ids
            for v in processed["insurance_policies"]["secondary_beneficiary"].values
        ]
    )
