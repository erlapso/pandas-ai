import os
import tempfile
import pytest
from pandasai.helpers.filemanager import DefaultFileManager

def test_default_file_manager_file_operations():
    """
    Test DefaultFileManager's file operations:
    - Check abs_path returns the correct absolute path.
    - Test writing and reading text files.
    - Test writing and reading binary files.
    - Test checking file/directory existence.
    - Test directory creation with mkdir.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override the base_path for testing purposes.
        manager = DefaultFileManager()
        manager.base_path = tmpdir
        # Test abs_path method.
        expected_path = os.path.join(tmpdir, "testfile.txt")
        assert manager.abs_path("testfile.txt") == expected_path
        # Write and load a text file.
        test_text = "Hello, world!"
        manager.write("testfile.txt", test_text)
        assert manager.exists("testfile.txt")
        loaded_text = manager.load("testfile.txt")
        assert loaded_text == test_text
        # Write and load a binary file.
        test_binary = b"binary content"
        manager.write_binary("testfile.bin", test_binary)
        loaded_binary = manager.load_binary("testfile.bin")
        assert loaded_binary == test_binary
        # Test directory creation.
        test_dir = "subdir"
        manager.mkdir(test_dir)
        # Verify that the new directory exists.
        assert os.path.isdir(manager.abs_path(test_dir))
def test_load_nonexistent_file_raises_error():
    """
    Test that attempting to load a non-existent file using DefaultFileManager
    raises a FileNotFoundError.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override the base_path for testing.
        manager = DefaultFileManager()
        manager.base_path = tmpdir
        # Test loading a non-existent text file raises FileNotFoundError.
        with pytest.raises(FileNotFoundError):
            manager.load("nonexistent.txt")
        # Test loading a non-existent binary file also raises FileNotFoundError.
        with pytest.raises(FileNotFoundError):
            manager.load_binary("nonexistent.bin")
def test_write_in_nonexistent_subdirectory_raises_error():
    """
    Test that writing a file in a non-existent subdirectory using DefaultFileManager 
    raises a FileNotFoundError. This ensures the file manager does not attempt to create 
    missing directories automatically for write operations.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override the base_path for testing.
        manager = DefaultFileManager()
        manager.base_path = tmpdir
        # Attempt to write a text file in a non-existent subdirectory.
        non_existent_text_path = os.path.join("nonexistent_dir", "file.txt")
        with pytest.raises(FileNotFoundError):
            manager.write(non_existent_text_path, "Some text content")
        # Attempt to write a binary file in a non-existent subdirectory.
        non_existent_binary_path = os.path.join("nonexistent_dir", "file.bin")
        with pytest.raises(FileNotFoundError):
            manager.write_binary(non_existent_binary_path, b"Some binary content")
def test_overwrite_file():
    """
    Test that writing to an existing file overwrites the previous content.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override the base_path for testing.
        manager = DefaultFileManager()
        manager.base_path = tmpdir
        file_name = "overwrite.txt"
        initial_content = "First version"
        updated_content = "Second version"
        # Write initial content to the file.
        manager.write(file_name, initial_content)
        # Verify that the file exists and the content is as expected.
        assert manager.exists(file_name)
        assert manager.load(file_name) == initial_content
        # Write updated content to the same file.
        manager.write(file_name, updated_content)
        # Check that the file content has been updated.
        assert manager.load(file_name) == updated_content
def test_load_text_file_with_invalid_encoding():
    """
    Test that loading a file that contains invalid UTF-8 data as text
    raises a UnicodeDecodeError.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = DefaultFileManager()
        manager.base_path = tmpdir
        # Write binary data that is not valid UTF-8.
        invalid_utf8_data = b"\xff"
        file_name = "invalid.txt"
        manager.write_binary(file_name, invalid_utf8_data)
        # Attempting to load the file as text should raise a UnicodeDecodeError.
        with pytest.raises(UnicodeDecodeError):
            manager.load(file_name)
def test_mkdir_on_existing_file_raises_error():
    """
    Test that attempting to create a directory on a path that already exists as a file
    raises a FileExistsError.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = DefaultFileManager()
        manager.base_path = tmpdir
        # Create a file at the target path.
        file_path = "existing"
        manager.write(file_path, "some content")
        
        # Attempting to use mkdir on a path where a file already exists should raise an error.
        with pytest.raises(FileExistsError):
            manager.mkdir(file_path)
def test_load_directory_raises_error():
    """
    Test that attempting to load a directory using DefaultFileManager.load()
    raises an IsADirectoryError.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override the base_path for testing
        manager = DefaultFileManager()
        manager.base_path = tmpdir
        
        # Create a directory using the file manager
        test_dir = "subdirectory"
        manager.mkdir(test_dir)
        
        # Attempting to load a directory should raise IsADirectoryError.
        # Note: Depending on the system python version, open() on a directory
        # may also raise an OSError. We use IsADirectoryError which is a subclass
        # of OSError for clarity.
        with pytest.raises(IsADirectoryError):
            manager.load(test_dir)
def test_empty_file_path_raises_error():
    """
    Test that passing an empty file path to load() and load_binary() results
    in an IsADirectoryError, since the empty file path resolves to the base directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override the base_path for testing purposes.
        manager = DefaultFileManager()
        manager.base_path = tmpdir
        # Attempting to load with an empty file path (i.e., the base directory)
        # should raise an IsADirectoryError for both text and binary file loading.
        with pytest.raises(IsADirectoryError):
            manager.load("")
        with pytest.raises(IsADirectoryError):
            manager.load_binary("")
def test_write_and_load_in_existing_nested_directory():
    """
    Test writing and reading a file in an existing nested directory.
    This ensures that if a subdirectory is explicitly created using mkdir(),
    then file writing and subsequent reading operations in that subdirectory work correctly.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override the base_path for testing.
        manager = DefaultFileManager()
        manager.base_path = tmpdir
        
        # Define a nested directory path and create it.
        nested_dir = os.path.join("nested", "subdir")
        manager.mkdir(nested_dir)
        
        # Define the file path inside the nested directory.
        file_path = os.path.join(nested_dir, "testfile.txt")
        content = "Content in nested directory"
        
        # Write the file in the nested directory.
        manager.write(file_path, content)
        
        # Verify that the file now exists and its contents match the expected content.
        assert manager.exists(file_path)
        loaded = manager.load(file_path)
        assert loaded == content
def test_load_with_absolute_file_path():
    """
    Test that loading a file using an absolute file path bypasses the base_path.
    This ensures that if an absolute file path is provided, the file manager's load
    method uses the absolute path directly and loads the correct file contents.
    """
    # Create a temporary file outside manager's base_path.
    with tempfile.NamedTemporaryFile('w+', delete=False, encoding='utf-8') as tmp_file:
        tmp_file.write("Absolute Test Content")
        tmp_file_name = tmp_file.name
    try:
        # Override the base_path for testing using a new temporary directory.
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DefaultFileManager()
            manager.base_path = tmpdir
            # When an absolute path is provided, os.path.join(self.base_path, file_path)
            # will ignore the base_path. Thus, the file should load from tmp_file_name.
            loaded_content = manager.load(tmp_file_name)
            assert loaded_content == "Absolute Test Content"
    finally:
        os.remove(tmp_file_name)