from gretel_trainer.benchmark import Datatype


def test_repo_list_tags(repo):
    tags = repo.list_tags()
    assert len(tags) > 0
    for expected_tag in ["small", "large", "e-commerce", "healthcare"]:
        assert expected_tag in tags


def test_repo_list_datasets(repo):
    all_datasets = repo.list_datasets()

    # querying by datatype, using enum or string
    tabular_datasets = repo.list_datasets(datatype="tabular")
    tabular_enum_datasets = repo.list_datasets(datatype=Datatype.TABULAR)
    assert len(tabular_datasets) > 0
    assert len(tabular_datasets) == len(tabular_enum_datasets)

    # querying by tag
    large_datasets = repo.list_datasets(tags=["large"])
    assert len(all_datasets) > len(large_datasets)

    # querying uses "and" logic
    small_tabular_datasets = repo.list_datasets(datatype="tabular", tags=["small"])
    assert len(small_tabular_datasets) > 0
    assert len(small_tabular_datasets) < len(tabular_datasets)

    assert len(repo.list_datasets(tags=["large", "small"])) == 0
