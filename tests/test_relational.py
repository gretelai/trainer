from gretel_trainer.relational.relationships import Relationships, TableProgress


def test_walk_ecommerce():
    relationships = Relationships() \
        .add(("events", "user_id"), ("users", "id")) \
        .add(("products", "distribution_center_id"), ("distribution_center", "id")) \
        .add(("inventory_items", "product_id"), ("products", "id")) \
        .add(("inventory_items", "distribution_center_id"), ("distribution_center", "id")) \
        .add(("order_items", "inventory_item_id"), ("inventory_items", "id")) \
        .add(("order_items", "user_id"), ("users", "id"))
    t = TableProgress(relationships)

    # To start, "eldest generation" tables (those with no parents / outbound foreign keys) are ready
    assert set(t.ready()) == {"users", "distribution_center"}

    t.mark_complete("users")

    # `events` was only blocked by `users`, and so now is ready
    assert set(t.ready()) == {"events", "distribution_center"}

    t.mark_complete("distribution_center")

    assert set(t.ready()) == {"events", "products"}

    t.mark_complete("events")
    t.mark_complete("products")

    assert set(t.ready()) == {"inventory_items"}

    t.mark_complete("inventory_items")

    assert set(t.ready()) == {"order_items"}

    t.mark_complete("order_items")

    assert set(t.ready()) == set()
