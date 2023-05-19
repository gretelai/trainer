import gretel_trainer.relational.ancestry as ancestry


def test_ecom_add_and_remove_ancestor_data(ecom):
    users_with_ancestors = ancestry.get_table_data_with_ancestors(ecom, "users")
    assert set(users_with_ancestors.columns) == {
        "self|id",
        "self|first_name",
        "self|last_name",
    }
    restored_users = ancestry.drop_ancestral_data(users_with_ancestors)
    assert set(restored_users.columns) == {"id", "first_name", "last_name"}

    events_with_ancestors = ancestry.get_table_data_with_ancestors(ecom, "events")
    assert set(events_with_ancestors.columns) == {
        "self|id",
        "self|browser",
        "self|traffic_source",
        "self|user_id",
        "self.user_id|id",
        "self.user_id|first_name",
        "self.user_id|last_name",
    }
    restored_events = ancestry.drop_ancestral_data(events_with_ancestors)
    assert set(restored_events.columns) == {
        "id",
        "browser",
        "traffic_source",
        "user_id",
    }

    inventory_items_with_ancestors = ancestry.get_table_data_with_ancestors(
        ecom, "inventory_items"
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
    restored_inventory_items = ancestry.drop_ancestral_data(
        inventory_items_with_ancestors
    )
    assert set(restored_inventory_items.columns) == {
        "id",
        "sold_at",
        "cost",
        "product_id",
        "product_distribution_center_id",
    }


def test_mutagenesis_add_and_remove_ancestor_data(mutagenesis):
    bond_with_ancestors = ancestry.get_table_data_with_ancestors(mutagenesis, "bond")
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
    restored_bond = ancestry.drop_ancestral_data(bond_with_ancestors)
    assert set(restored_bond.columns) == {"type", "atom1_id", "atom2_id"}


def test_tpch_add_and_remove_ancestor_data(tpch):
    lineitem_with_ancestors = ancestry.get_table_data_with_ancestors(tpch, "lineitem")
    assert set(lineitem_with_ancestors.columns) == {
        "self|l_partkey",
        "self|l_suppkey",
        "self|l_quantity",
        "self.l_partkey+l_suppkey|ps_partkey",
        "self.l_partkey+l_suppkey|ps_suppkey",
        "self.l_partkey+l_suppkey|ps_availqty",
        "self.l_partkey+l_suppkey.ps_partkey|p_partkey",
        "self.l_partkey+l_suppkey.ps_partkey|p_name",
        "self.l_partkey+l_suppkey.ps_suppkey|s_suppkey",
        "self.l_partkey+l_suppkey.ps_suppkey|s_name",
    }
    restored_lineitem = ancestry.drop_ancestral_data(lineitem_with_ancestors)
    assert set(restored_lineitem.columns) == {"l_partkey", "l_suppkey", "l_quantity"}


def test_ancestral_data_from_different_tablesets(source_nba, synthetic_nba):
    source_nba, _, _, _ = source_nba
    _, custom_states, custom_cities, custom_teams = synthetic_nba

    # By default, get data from source
    source_teams_with_ancestors = ancestry.get_table_data_with_ancestors(
        source_nba, "teams"
    )
    assert set(source_teams_with_ancestors["self|name"]) == {"Lakers", "Grizzlies"}

    custom_tableset = {
        "states": custom_states,
        "cities": custom_cities,
        "teams": custom_teams,
    }

    # Optionally provide a different tableset
    custom_teams_with_ancestors = ancestry.get_table_data_with_ancestors(
        source_nba, "teams", custom_tableset
    )
    assert set(custom_teams_with_ancestors["self|name"]) == {"Sixers", "Heat"}


def test_whether_column_is_ancestral(mutagenesis):
    assert ancestry.is_ancestral_column("self|atom1_id") is False
    assert ancestry.is_ancestral_column("self.atom1_id|atom1_id")
    assert ancestry.is_ancestral_column("self.atom1_id.molecule_id|atom1_id")


def test_primary_key_in_multigenerational_format(mutagenesis):
    assert ancestry.get_multigenerational_primary_key(mutagenesis, "bond") == [
        "self|atom1_id",
        "self|atom2_id",
    ]
    assert ancestry.get_multigenerational_primary_key(mutagenesis, "atom") == [
        "self|atom_id"
    ]


def test_ancestral_foreign_key_maps(ecom):
    events_afk_maps = ancestry.get_ancestral_foreign_key_maps(ecom, "events")
    assert events_afk_maps == [("self|user_id", "self.user_id|id")]

    inventory_items_afk_maps = ancestry.get_ancestral_foreign_key_maps(
        ecom, "inventory_items"
    )
    assert ("self|product_id", "self.product_id|id") in inventory_items_afk_maps
    assert (
        "self|product_distribution_center_id",
        "self.product_distribution_center_id|id",
    ) in inventory_items_afk_maps


def test_ancestral_foreign_key_maps_composite(tpch):
    lineitem_afk_maps = ancestry.get_ancestral_foreign_key_maps(tpch, "lineitem")
    assert (
        "self|l_partkey",
        "self.l_partkey+l_suppkey|ps_partkey",
    ) in lineitem_afk_maps
    assert (
        "self|l_suppkey",
        "self.l_partkey+l_suppkey|ps_suppkey",
    ) in lineitem_afk_maps


def test_prepend_foreign_key_lineage(ecom):
    multigen_inventory_items = ancestry.get_table_data_with_ancestors(
        ecom, "inventory_items"
    )
    order_items_parent_data = ancestry.prepend_foreign_key_lineage(
        multigen_inventory_items, ["inventory_item_id"]
    )
    assert set(order_items_parent_data.columns) == {
        "self.inventory_item_id|id",
        "self.inventory_item_id|sold_at",
        "self.inventory_item_id|cost",
        "self.inventory_item_id|product_id",
        "self.inventory_item_id|product_distribution_center_id",
        "self.inventory_item_id.product_id|id",
        "self.inventory_item_id.product_id|name",
        "self.inventory_item_id.product_id|brand",
        "self.inventory_item_id.product_id|distribution_center_id",
        "self.inventory_item_id.product_distribution_center_id|id",
        "self.inventory_item_id.product_distribution_center_id|name",
        "self.inventory_item_id.product_id.distribution_center_id|id",
        "self.inventory_item_id.product_id.distribution_center_id|name",
    }


def test_get_seed_safe_multigenerational_columns_1(pets):
    table_cols = ancestry.get_seed_safe_multigenerational_columns(pets)

    expected = {
        "humans": {"self|id", "self|name", "self|city"},
        "pets": {
            "self|id",
            "self|name",
            "self|age",
            "self|human_id",
            "self.human_id|id",
            # "self.human_id|name", # highly unique categorical
            "self.human_id|city",
        },
    }

    assert set(table_cols.keys()) == set(expected.keys())
    for table, expected_cols in expected.items():
        assert set(table_cols[table]) == expected_cols


def test_get_seed_safe_multigenerational_columns_2(source_nba):
    source_nba = source_nba[0]
    table_cols = ancestry.get_seed_safe_multigenerational_columns(source_nba)

    expected = {
        "teams": {
            "self|name",
            "self|id",
            "self|city_id",
            "self.city_id|id",
            "self.city_id|state_id",
            # "self.city_id|name", # highly unique categorical
            "self.city_id.state_id|id",
            # "self.city_id.state_id|name", # highly unique categorical
        },
        "cities": {
            "self|id",
            "self|state_id",
            "self|name",
            "self.state_id|id",
            # "self.state_id|name", # highly unique categorical
        },
        "states": {
            "self|id",
            "self|name",
        },
    }

    assert set(table_cols.keys()) == set(expected.keys())
    for table, expected_cols in expected.items():
        assert set(table_cols[table]) == expected_cols
