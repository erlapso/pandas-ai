import pytest
from types import SimpleNamespace
from pandasai.data_loader.local_loader import LocalDatasetLoader
from pandasai.exceptions import InvalidDataSourceType

def test_invalid_source_type_raises_invalid_data_source_type():
    """Test that using an unsupported local source type raises InvalidDataSourceType."""
    # Create a fake schema with an unsupported source type ("json")
    fake_schema = SimpleNamespace(
        name="dummy",
        source=SimpleNamespace(type="json", path="dummy.json"),
        columns=[],  # No columns defined
        group_by=None
    )

    dataset_path = "org/dataset"
    loader = LocalDatasetLoader(fake_schema, dataset_path)

    with pytest.raises(InvalidDataSourceType):
        loader.load()