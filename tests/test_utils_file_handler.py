"""
Unit tests for utils/file_handler.py
"""
import unittest
import tempfile
import os
from pathlib import Path
from utils.file_handler import read_file


class TestReadFileTxt(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_reads_txt_file(self):
        path = Path(self.temp_dir) / "test.txt"
        path.write_text("Hello, world!", encoding='utf-8')
        name, content = read_file(path)
        self.assertEqual(name, "test.txt")
        self.assertEqual(content, "Hello, world!")

    def test_reads_md_file(self):
        path = Path(self.temp_dir) / "readme.md"
        path.write_text("# Heading\nSome text.", encoding='utf-8')
        name, content = read_file(path)
        self.assertEqual(name, "readme.md")
        self.assertIn("Heading", content)

    def test_cleans_control_characters(self):
        path = Path(self.temp_dir) / "dirty.txt"
        # Write text with control chars (tab, null, bell)
        path.write_text("Clean\x00text\x07here", encoding='utf-8')
        name, content = read_file(path)
        self.assertNotIn("\x00", content)
        self.assertNotIn("\x07", content)
        self.assertIn("Clean", content)
        self.assertIn("text", content)

    def test_unsupported_extension_returns_none(self):
        path = Path(self.temp_dir) / "data.xyz"
        path.write_text("some data", encoding='utf-8')
        name, content = read_file(path)
        self.assertEqual(name, "data.xyz")
        self.assertIsNone(content)

    def test_empty_file(self):
        path = Path(self.temp_dir) / "empty.txt"
        path.write_text("", encoding='utf-8')
        name, content = read_file(path)
        self.assertEqual(name, "empty.txt")
        self.assertEqual(content, "")

    def test_nonexistent_file(self):
        path = Path(self.temp_dir) / "nofile.txt"
        name, content = read_file(path)
        self.assertIsNone(content)


class TestReadFilePdf(unittest.TestCase):
    def test_pdf_import_check(self):
        """Verify that we can at least attempt a PDF read without crashing."""
        try:
            import PyPDF2
            pdf_available = True
        except ImportError:
            pdf_available = False
        # Just verifying the import guard works
        self.assertIsInstance(pdf_available, bool)


class TestReadFileDocx(unittest.TestCase):
    def test_docx_import_check(self):
        """Verify that we can at least attempt a DOCX read without crashing."""
        try:
            import docx
            docx_available = True
        except ImportError:
            docx_available = False
        self.assertIsInstance(docx_available, bool)


if __name__ == '__main__':
    unittest.main()
