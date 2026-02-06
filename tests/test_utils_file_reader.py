"""
Unit tests for utils/file_reader.py
"""
import unittest
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock


class TestReadFiles(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch('utils.file_reader.SimpleDirectoryReader')
    def test_read_files_with_valid_files(self, mock_reader_cls):
        """Test that read_files processes valid file paths."""
        # Create a real file so os.path.isfile returns True
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("Hello, test content")

        mock_doc = MagicMock()
        mock_doc.text = "Hello, test content"
        mock_doc.metadata = {"file_name": "test.txt"}

        mock_reader_instance = MagicMock()
        mock_reader_instance.load_data.return_value = [mock_doc]
        mock_reader_cls.return_value = mock_reader_instance

        from utils.file_reader import read_files
        result = read_files([test_file])

        self.assertEqual(len(result), 1)
        # Content should have been cleaned (no control chars)
        self.assertNotIn("\x00", result[0].text)

    def test_read_files_with_no_paths(self):
        """Test that read_files returns empty for non-existent paths."""
        from utils.file_reader import read_files
        result = read_files(["/nonexistent/path/file.txt"])
        self.assertEqual(len(result), 0)

    def test_read_files_empty_list(self):
        """Test that read_files handles empty input."""
        from utils.file_reader import read_files
        result = read_files([])
        self.assertEqual(len(result), 0)


if __name__ == '__main__':
    unittest.main()
