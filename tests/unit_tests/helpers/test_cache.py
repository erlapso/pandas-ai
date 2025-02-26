import uuid

import tempfile
from pandasai.core.cache import Cache
import pytest

class TestCache:
    def test_abs_path(self):
        """Test that the Cache uses the provided absolute path and its files are placed in the given directory."""
        import os, uuid, glob
        # Create a temporary directory and use it as the cache directory.
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_filename = f"cache_{uuid.uuid4().hex}"
            cache = Cache(filename=cache_filename, abs_path=tmp_dir)
            # Verify that the cache file is created in the provided temporary directory.
            assert os.path.dirname(cache.filepath) == tmp_dir
            # Set and get a value.
            cache.set("abs_key", "abs_value")
            assert cache.get("abs_key") == "abs_value"
            # The primary cache file should exist.
            assert os.path.exists(cache.filepath)
            # Destroy the cache.
            cache.destroy()
            # Verify that any additional duckdb-generated files (with extra extensions) are removed.
            assert glob.glob(f"{cache.filepath}.*") == []
    def test_cache(self):
        cache_filename = f"cache_{uuid.uuid4().hex}"
        cache = Cache(filename=cache_filename)
        cache.set("key", "value")
        assert cache.get("key") == "value"

        cache.delete("key")
        print(cache.get("key"))
        assert cache.get("key") is None

        cache.destroy()

    def test_read_cache(self):
        cache_filename = f"cache_{uuid.uuid4().hex}"
        cache = Cache(filename=cache_filename)
        cache.set("key", "value")
        cache.close()

        cache = Cache(filename=cache_filename)
        assert cache.get("key") == "value"

        cache.destroy()

    def test_delete_duplicate_keys(self):
        """Test that setting the same key twice allows deletion to remove all entries for that key."""
        import glob, os
        cache_filename = f"cache_{uuid.uuid4().hex}"
        cache = Cache(filename=cache_filename)
        # Set the same key twice with different values.
        cache.set("dup_key", "first_value")
        cache.set("dup_key", "second_value")
        # Retrieve the key; expect one of the values to be returned (order is not guaranteed)
        result = cache.get("dup_key")
        assert result in ["first_value", "second_value"]
        # Delete the key and verify that get returns None.
        cache.delete("dup_key")
        assert cache.get("dup_key") is None
        cache.destroy()
    def test_get_cache_key(self):
        """Test that get_cache_key returns a unique key composed of the conversation and dfs schema names."""
        cache_filename = f"cache_{uuid.uuid4().hex}"
        cache = Cache(filename=cache_filename)
    
        # Create a dummy context with a memory object and a list of dummy dfs objects
        class DummyMemory:
            def get_conversation(self):
                return "conversation_"
    
        class DummySchema:
            def __init__(self, name):
                self.name = name
    
        class DummyDF:
            def __init__(self, name):
                self.schema = DummySchema(name)
    
        class DummyContext:
            pass
    
        dummy_context = DummyContext()
        dummy_context.memory = DummyMemory()
        dummy_context.dfs = [DummyDF("dfA"), DummyDF("dfB")]
    
        expected_key = "conversation_dfAdfB"
        result = cache.get_cache_key(dummy_context)
        assert result == expected_key
    
        cache.destroy()
    def test_clear_cache(self):
        """Test that clear() removes all entries from the cache."""
        cache_filename = f"cache_{uuid.uuid4().hex}"
        cache = Cache(filename=cache_filename)
        # Set multiple key-value pairs in the cache.
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Verify that keys exist before clearing.
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"

        # Clear the cache and verify that keys are removed.
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

        cache.destroy()
    def test_versioned_key(self):
        """Test that versioned_key returns the correct versioned key using CACHE_TOKEN."""
        from pandasai.constants import CACHE_TOKEN
        cache_filename = f"cache_{uuid.uuid4().hex}"
        cache = Cache(filename=cache_filename)
        expected_key = f"{CACHE_TOKEN}-sampleKey"
        assert cache.versioned_key("sampleKey") == expected_key
        cache.destroy()
    def test_get_cache_key_empty_dfs(self):
        """Test that get_cache_key returns only the conversation string when dfs is empty."""
        cache_filename = f"cache_{uuid.uuid4().hex}"
        cache = Cache(filename=cache_filename)
        # Create a dummy context with a memory object and an empty list for dfs.
        class DummyMemory:
            def get_conversation(self):
                return "conversation_only"
        class DummyContext:
            pass
        dummy_context = DummyContext()
        dummy_context.memory = DummyMemory()
        dummy_context.dfs = []
        expected_key = "conversation_only"
        result = cache.get_cache_key(dummy_context)
        assert result == expected_key
        cache.destroy()
    def test_set_non_string_key_value(self):
        """Test that using non-string key and value automatically converts them to strings."""
        cache_filename = f"cache_{uuid.uuid4().hex}"
        cache = Cache(filename=cache_filename)
        # Using a numeric key and a float value
        key = 42
        value = 3.14159
        cache.set(key, value)
        retrieved = cache.get(key)
        assert retrieved == str(value)
        cache.destroy()
    def test_operation_after_close(self):
        """Test that operations on a closed cache raise an exception and that destroy still cleans up files."""
        cache_filename = f"cache_{uuid.uuid4().hex}"
        cache = Cache(filename=cache_filename)
        cache.set("key", "value")
        cache.close()
        with pytest.raises(Exception):
            _ = cache.get("key")
        # Even after closing, destroy should clean up cache files without raising an exception
        cache.destroy()
        import glob
        assert glob.glob(f"{cache.filepath}.*") == []