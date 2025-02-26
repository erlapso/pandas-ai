import pytest
from pydantic import ValidationError

from pandasai.data_loader.semantic_layer_schema import (
    Destination,
    SemanticLayerSchema,
    Transformation,
    is_schema_source_same,
)


class TestSemanticLayerSchema:
    def test_valid_schema(self, raw_sample_schema):
        schema = SemanticLayerSchema(**raw_sample_schema)

        assert schema.name == "Users"
        assert schema.update_frequency == "weekly"
        assert len(schema.columns) == 3
        assert schema.order_by == ["created_at DESC"]
        assert schema.limit == 100
        assert len(schema.transformations) == 2
        assert schema.source.type == "csv"

    def test_valid_raw_mysql_schema(self, raw_mysql_schema):
        schema = SemanticLayerSchema(**raw_mysql_schema)

        assert schema.name == "users"
        assert schema.update_frequency == "weekly"
        assert len(schema.columns) == 3
        assert schema.order_by == ["created_at DESC"]
        assert schema.limit == 100
        assert len(schema.transformations) == 2
        assert schema.source.type == "mysql"

    def test_valid_raw_mysql_view_schema(self, raw_mysql_view_schema):
        schema = SemanticLayerSchema(**raw_mysql_view_schema)

        assert schema.name == "parent_children"
        assert len(schema.columns) == 3
        assert schema.view == True

    def test_missing_source_path(self, raw_sample_schema):
        raw_sample_schema["source"].pop("path")

        with pytest.raises(ValidationError):
            SemanticLayerSchema(**raw_sample_schema)

    def test_missing_source_table(self, raw_mysql_schema):
        raw_mysql_schema["source"].pop("table")

        with pytest.raises(ValidationError):
            SemanticLayerSchema(**raw_mysql_schema)

    def test_missing_mysql_connection(self, raw_mysql_schema):
        raw_mysql_schema["source"].pop("connection")

        with pytest.raises(ValidationError):
            SemanticLayerSchema(**raw_mysql_schema)

    def test_invalid_schema_missing_name(self, raw_sample_schema):
        raw_sample_schema.pop("name")

        with pytest.raises(ValidationError):
            SemanticLayerSchema(**raw_sample_schema)

    def test_invalid_column_type(self, raw_sample_schema):
        raw_sample_schema["columns"][0]["type"] = "unsupported"

        with pytest.raises(ValidationError):
            SemanticLayerSchema(**raw_sample_schema)

    def test_invalid_source_type(self, raw_sample_schema):
        raw_sample_schema["source"]["type"] = "invalid"

        with pytest.raises(ValidationError):
            SemanticLayerSchema(**raw_sample_schema)

    def test_valid_transformations(self):
        transformation_data = {
            "type": "anonymize",
            "params": {"column": "email"},
        }

        transformation = Transformation(**transformation_data)

        assert transformation.type == "anonymize"
        assert transformation.params.column == "email"

    def test_valid_destination(self):
        destination_data = {
            "type": "local",
            "format": "parquet",
            "path": "output.parquet",
        }

        destination = Destination(**destination_data)

        assert destination.type == "local"
        assert destination.format == "parquet"
        assert destination.path == "output.parquet"

    def test_invalid_destination_format(self):
        destination_data = {
            "type": "local",
            "format": "invalid",
            "path": "output.parquet",
        }

        with pytest.raises(ValidationError):
            Destination(**destination_data)

    def test_invalid_transformation_type(self):
        transformation_data = {
            "type": "unsupported_transformation",
            "params": {"column": "email"},
        }

        with pytest.raises(ValidationError):
            Transformation(**transformation_data)

    def test_is_schema_source_same_true(self, raw_mysql_schema):
        schema1 = SemanticLayerSchema(**raw_mysql_schema)
        schema2 = SemanticLayerSchema(**raw_mysql_schema)

        assert is_schema_source_same(schema1, schema2) is True

    def test_is_schema_source_same_false(self, raw_mysql_schema, raw_sample_schema):
        schema1 = SemanticLayerSchema(**raw_mysql_schema)
        schema2 = SemanticLayerSchema(**raw_sample_schema)

        assert is_schema_source_same(schema1, schema2) is False

    def test_invalid_view_and_source(self, raw_mysql_schema):
        raw_mysql_schema["view"] = True

        with pytest.raises(ValidationError):
            SemanticLayerSchema(**raw_mysql_schema)

    def test_invalid_source_missing_view_or_table(self, raw_mysql_schema):
        raw_mysql_schema["source"].pop("table")

        with pytest.raises(ValidationError):
            SemanticLayerSchema(**raw_mysql_schema)

    def test_invalid_no_relation_for_view(self, raw_mysql_view_schema):
        raw_mysql_view_schema.pop("relations")

        with pytest.raises(ValidationError):
            SemanticLayerSchema(**raw_mysql_view_schema)

    def test_invalid_duplicated_columns(self, raw_sample_schema):
        raw_sample_schema["columns"].append(raw_sample_schema["columns"][0])

        with pytest.raises(ValidationError):
            SemanticLayerSchema(**raw_sample_schema)

    def test_invalid_wrong_column_format_in_view(self, raw_mysql_view_schema):
        raw_mysql_view_schema["columns"][0]["name"] = "parentsid"

        with pytest.raises(ValidationError):
            SemanticLayerSchema(**raw_mysql_view_schema)

    def test_invalid_uncovered_columns_in_view(self, raw_mysql_view_schema):
        """Test that a view with uncovered tables in the columns (i.e. missing relations for some tables) raises a ValueError."""
        # Force the schema to have multiple tables in columns by ensuring relations are empty
        raw_mysql_view_schema["relations"] = []
        with pytest.raises(ValueError, match="No relations provided for the following tables"):
            SemanticLayerSchema(**raw_mysql_view_schema)
    
    def test_invalid_rename_missing_new_name(self, raw_sample_schema):
        """Test that a rename transformation without 'new_name' parameter raises a ValidationError."""
        transformation_data = {
            "type": "rename",
            "params": {"column": "username"}  # Note: missing 'new_name'
        }
        with pytest.raises(ValidationError):
            Transformation(**transformation_data)
        raw_sample_schema["columns"][0]["name"] = "parents.id"

        with pytest.raises(ValidationError):
            SemanticLayerSchema(**raw_sample_schema)

    def test_invalid_wrong_relation_format_in_view(self, raw_mysql_view_schema):
        raw_mysql_view_schema["relations"][0]["to"] = "parentsid"

        with pytest.raises(ValidationError):
            SemanticLayerSchema(**raw_mysql_view_schema)

    def test_invalid_group_by_missing_columns(self, raw_sample_schema, raw_mysql_view_schema):
        """Test that the schema fails when group_by is provided but not all non-aggregated columns are included."""
        # Assume raw_sample_schema is a valid table schema with columns that do not have an aggregation expression.
        # Setting group_by with only the first column, leaving the others unmatched.
        raw_sample_schema["group_by"] = [raw_sample_schema["columns"][0]["name"]]
        with pytest.raises(ValidationError):
            SemanticLayerSchema(**raw_sample_schema)
        raw_mysql_view_schema["relations"][0]["to"] = "parents.id"

        with pytest.raises(ValidationError):
            SemanticLayerSchema(**raw_mysql_view_schema)

    def test_sql_connection_config_equality(self):
        """Test that SQLConnectionConfig equality works correctly."""
        from pandasai.data_loader.semantic_layer_schema import SQLConnectionConfig
        config1 = SQLConnectionConfig(host="localhost", port=3306, database="test_db", user="user", password="pass")
        config2 = SQLConnectionConfig(host="localhost", port=3306, database="test_db", user="user", password="pass")
        config3 = SQLConnectionConfig(host="localhost", port=3306, database="test_db", user="user", password="different")
        assert config1 == config2
        assert config1 != config3
    def test_to_dict_and_to_yaml(self, raw_sample_schema):
        """Test that the schema's to_dict and to_yaml methods produce correct outputs without None values."""
        schema = SemanticLayerSchema(**raw_sample_schema)
        schema_dict = schema.to_dict()
        yaml_output = schema.to_yaml()
        # Ensure that dictionary output does not include None values
        for key, value in schema_dict.items():
            assert value is not None
        # Check that the YAML output contains some expected key (e.g., 'name')
        assert "name:" in yaml_output
        # Verify that loading the YAML output produces the same dictionary
        import yaml
        loaded_yaml = yaml.safe_load(yaml_output)
        assert loaded_yaml == schema_dict
    def test_invalid_view_format_in_table(self, raw_sample_schema):
        """Test that a table schema with view-formatted column names raises an error."""
        # Modify raw_sample_schema columns to use view format (e.g., "dataset.column") even though this is a table schema
        for col in raw_sample_schema["columns"]:
            col["name"] = f"dataset.{col['name']}"
        with pytest.raises(ValidationError, match="All columns in a table must be in the format '\\[column\\]'."):
            SemanticLayerSchema(**raw_sample_schema)