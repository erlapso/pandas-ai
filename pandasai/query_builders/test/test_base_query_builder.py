import pytest
from pandasai.query_builders.base_query_builder import BaseQueryBuilder

class DummySource:
    def __init__(self, value):
        self.value = value

    def is_compatible_source(self, other):
        return self.value == other.value

def test_check_compatible_sources_incompatible():
    """Test that check_compatible_sources returns False when given incompatible sources."""
    # Create one source that will be used as the base and another that is incompatible
    source_compatible = DummySource(1)
    source_incompatible = DummySource(2)
    sources = [source_compatible, source_incompatible]

    result = BaseQueryBuilder.check_compatible_sources(sources)
    assert result is False
def test_check_compatible_sources_compatible():
    """Test that check_compatible_sources returns True when all sources are compatible."""
    source1 = DummySource(1)
    source2 = DummySource(1)
    sources = [source1, source2]

    result = BaseQueryBuilder.check_compatible_sources(sources)
    assert result is True
def test_build_query_full_options():
    """Test that build_query generates a query with group by, order by, limit, and alias correctly."""
    # Dummy classes to simulate the SemanticLayerSchema and Column objects
    class DummyColumn:
        def __init__(self, name, expression=None, alias=None):
            self.name = name
            self.expression = expression
            self.alias = alias
    
    class DummySemanticLayerSchema:
        def __init__(self):
            self.name = "my_table"
            self.columns = [
                DummyColumn("col1", alias="c1"),
                DummyColumn("col2", expression="SUM(col2)")
            ]
            self.group_by = ["col1"]
            self.order_by = ["col1"]
            self.limit = 10
    
    dummy_schema = DummySemanticLayerSchema()
    qb = BaseQueryBuilder(schema=dummy_schema)
    query = qb.build_query()
    # Check that the query string contains GROUP BY, ORDER BY, and LIMIT clauses.
    assert "GROUP BY" in query, "Expected 'GROUP BY' in the generated SQL query."
    assert "ORDER BY" in query, "Expected 'ORDER BY' in the generated SQL query."
    assert "LIMIT" in query, "Expected 'LIMIT' in the generated SQL query."
    # Check that the alias is correctly applied (should contain 'AS c1')
    assert "AS c1" in query, "Expected column alias 'AS c1' in the query."
def test_get_row_count():
    """Test that get_row_count returns a valid count query SQL."""
    class DummySemanticLayerSchema:
        def __init__(self):
            self.name = "test_table"
            self.columns = []
            self.group_by = None
            self.order_by = None
            self.limit = None
    
    dummy_schema = DummySemanticLayerSchema()
    qb = BaseQueryBuilder(schema=dummy_schema)
    query = qb.get_row_count()
    assert "COUNT(*)" in query, "Expected COUNT(*) in the count query."
    assert "test_table" in query, "Expected table name 'test_table' in the count query."
def test_get_head_query():
    """Test that get_head_query generates a query using the provided limit and includes GROUP BY but excludes ORDER BY."""
    class DummyColumn:
        def __init__(self, name, expression=None, alias=None):
            self.name = name
            self.expression = expression
            self.alias = alias

    class DummySemanticLayerSchema:
        def __init__(self):
            self.name = "dummy_table"
            self.columns = [DummyColumn("col1")]
            self.group_by = ["col1"]
            self.order_by = ["col1"]  # even if order_by is set, get_head_query should ignore it
            self.limit = None

    dummy_schema = DummySemanticLayerSchema()
    qb = BaseQueryBuilder(schema=dummy_schema)
    query = qb.get_head_query(7)

    # Check that GROUP BY clause is present.
    assert "GROUP BY" in query, "Expected 'GROUP BY' in the generated head query."

    # Check that ORDER BY clause is not present in get_head_query output.
    assert "ORDER BY" not in query, "Did not expect 'ORDER BY' in the head query."

    # Check that the LIMIT clause is correctly applied.
    assert "LIMIT 7" in query, "Expected 'LIMIT 7' in the generated head query."
def test_build_query_with_no_columns():
    """Test that build_query returns a query with default '*' when no columns, group_by, order_by, or limit are provided."""
    class DummySemanticLayerSchema:
        def __init__(self):
            self.name = "empty_table"
            self.columns = []
            self.group_by = None
            self.order_by = None
            self.limit = None

    dummy_schema = DummySemanticLayerSchema()
    qb = BaseQueryBuilder(schema=dummy_schema)
    query = qb.build_query()
    normalized_query = " ".join(query.split())
    assert "SELECT *" in normalized_query, "Expected query to select all columns with '*'"
    assert "FROM" in normalized_query and "empty_table" in normalized_query, "Expected table name 'empty_table' in the query"
    # Ensure group by, order by, and limit clauses are not present
    assert "GROUP BY" not in normalized_query, "Did not expect 'GROUP BY' in the query"
    assert "ORDER BY" not in normalized_query, "Did not expect 'ORDER BY' in the query"
    assert "LIMIT" not in normalized_query, "Did not expect 'LIMIT' in the query"
def test_get_head_query_no_group_by():
    """Test that get_head_query does not include GROUP BY or ORDER BY when group_by is not provided, even if order_by is set."""
    class DummyColumn:
        def __init__(self, name, expression=None, alias=None):
            self.name = name
            self.expression = expression
            self.alias = alias

    class DummySemanticLayerSchema:
        def __init__(self):
            self.name = "no_group_table"
            self.columns = [DummyColumn("col1")]
            self.group_by = None
            self.order_by = ["col1"]  # even if order_by is provided, it should be ignored in get_head_query
            self.limit = None

    dummy_schema = DummySemanticLayerSchema()
    qb = BaseQueryBuilder(schema=dummy_schema)
    query = qb.get_head_query(3)
    normalized_query = " ".join(query.split())
    # Assert that there is no GROUP BY or ORDER BY clause in the generated head query.
    assert "GROUP BY" not in normalized_query, "Did not expect 'GROUP BY' in the head query when not provided."
    assert "ORDER BY" not in normalized_query, "Did not expect 'ORDER BY' in the head query, even if set in schema."
    # Assert that LIMIT clause is correctly applied.
    assert "LIMIT 3" in normalized_query, "Expected 'LIMIT 3' in the head query."
    # Assert that the table name is present.
    assert "no_group_table" in normalized_query, "Expected table name 'no_group_table' in the head query."
def test_check_compatible_sources_single():
    """Test that check_compatible_sources returns True when only one source is provided."""
    single_source = DummySource(42)
    result = BaseQueryBuilder.check_compatible_sources([single_source])
    assert result is True