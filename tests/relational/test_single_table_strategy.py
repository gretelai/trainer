from gretel_trainer.relational.strategies.single_table import SingleTableStrategy


def test_prepare_training_data_removes_primary_and_foreign_keys(pets):
    strategy = SingleTableStrategy()

    training_pets = strategy.prepare_training_data("pets", pets)

    assert set(training_pets.columns) == {"name", "age"}


def test_retraining_a_set_of_tables_only_retrains_those_tables(ecom):
    strategy = SingleTableStrategy()
    assert set(strategy.tables_to_retrain(["users"], ecom)) == {"users"}
    assert set(strategy.tables_to_retrain(["users", "events"], ecom)) == {
        "users",
        "events",
    }
    assert set(strategy.tables_to_retrain(["products"], ecom)) == {"products"}


def test_table_generation_readiness(ecom):
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
