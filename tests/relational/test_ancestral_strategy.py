from gretel_trainer.relational.strategies.ancestral import AncestralStrategy


def test_prepare_training_data_returns_multigenerational_data_without_keys_or_highly_unique_categorial_fields(
    pets,
):
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


def test_retraining_a_set_of_tables_forces_retraining_descendants_as_well(ecom):
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


def test_table_generation_readiness(ecom):
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
