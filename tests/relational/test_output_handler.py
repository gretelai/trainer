def test_uploads_path_to_project_and_stores_artifact_key(output_handler, pets):
    project = output_handler._project
    project.upload_artifact.return_value = "artifact_key"

    output_handler.save_sources(pets)

    project.upload_artifact.assert_called_once()
    assert output_handler.get_source_archive() == "artifact_key"


def test_overwrites_project_artifacts(output_handler, pets):
    output_handler._source_archive = "first_key"

    project = output_handler._project
    project.upload_artifact.return_value = "second_key"

    output_handler.save_sources(pets)

    project.upload_artifact.assert_called_once()
    project.delete_artifact.assert_called_once_with("first_key")
    assert output_handler.get_source_archive() == "second_key"
