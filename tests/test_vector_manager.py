"""
Unit tests for core/vector_manager.py

NOTE: VectorManager imports chromadb which requires torch.
If torch is broken in the environment, heavy tests will be skipped.
"""
import unittest
import tempfile
import os
import shutil
import json
from pathlib import Path

# Check if the heavy imports work
try:
    from unittest.mock import patch, MagicMock
    from core.vector_manager import get_file_hash
    VECTOR_IMPORT_OK = True
except (ImportError, OSError) as e:
    VECTOR_IMPORT_OK = False
    VECTOR_IMPORT_ERROR = str(e)


class TestVectorManagerHelpers(unittest.TestCase):
    @unittest.skipUnless(VECTOR_IMPORT_OK, "vector_manager import failed (torch/chromadb)")
    def test_get_file_hash_consistent(self):
        """Test that file hashing produces consistent results."""
        temp_dir = tempfile.mkdtemp()
        try:
            path = os.path.join(temp_dir, "test.txt")
            with open(path, 'w') as f:
                f.write("Hello, world!")
            h1 = get_file_hash(path)
            h2 = get_file_hash(path)
            self.assertEqual(h1, h2)
            self.assertEqual(len(h1), 64)  # SHA-256 hex length
        finally:
            shutil.rmtree(temp_dir)

    @unittest.skipUnless(VECTOR_IMPORT_OK, "vector_manager import failed (torch/chromadb)")
    def test_get_file_hash_different_content(self):
        """Test that different files produce different hashes."""
        temp_dir = tempfile.mkdtemp()
        try:
            path1 = os.path.join(temp_dir, "a.txt")
            path2 = os.path.join(temp_dir, "b.txt")
            with open(path1, 'w') as f:
                f.write("Content A")
            with open(path2, 'w') as f:
                f.write("Content B")
            h1 = get_file_hash(path1)
            h2 = get_file_hash(path2)
            self.assertNotEqual(h1, h2)
        finally:
            shutil.rmtree(temp_dir)


class TestVectorManagerMetadata(unittest.TestCase):
    """Test metadata load/save operations in isolation (no heavy deps)."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_load_metadata_empty(self):
        """Test that missing metadata file means no data."""
        metadata_path = os.path.join(self.temp_dir, 'corpus_metadata.json')
        self.assertFalse(os.path.exists(metadata_path))

    def test_save_and_load_metadata(self):
        """Test round-trip save and load of metadata."""
        metadata_path = os.path.join(self.temp_dir, 'corpus_metadata.json')
        test_data = {"/path/to/file.txt": {"hash": "abc123", "doc_type": "document"}}

        with open(metadata_path, 'w') as f:
            json.dump(test_data, f, indent=4)

        with open(metadata_path, 'r') as f:
            loaded = json.load(f)

        self.assertEqual(loaded, test_data)

    def test_metadata_structure(self):
        """Test expected metadata structure."""
        test_data = {
            "/path/corpus/abc-123.txt": {
                "original_path": "/home/user/interview.txt",
                "doc_type": "interview",
                "hash": "sha256hex",
                "added_at": "2024-01-01T00:00:00+00:00"
            }
        }
        metadata_path = os.path.join(self.temp_dir, 'corpus_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(test_data, f)
        with open(metadata_path, 'r') as f:
            loaded = json.load(f)
        entry = loaded["/path/corpus/abc-123.txt"]
        self.assertIn("original_path", entry)
        self.assertIn("doc_type", entry)
        self.assertIn("hash", entry)
        self.assertIn("added_at", entry)


@unittest.skipUnless(VECTOR_IMPORT_OK, "vector_manager import failed (torch/chromadb)")
class TestVectorManagerInit(unittest.TestCase):
    """Test VectorManager initialization with mocked dependencies."""

    @patch('core.vector_manager.get_chroma_client')
    @patch('core.vector_manager.get_embed_model')
    @patch('core.vector_manager.VectorStoreIndex')
    @patch('core.vector_manager.StorageContext')
    @patch('core.vector_manager.ChromaVectorStore')
    @patch('core.vector_manager.LlamaSettings')
    def test_init_with_valid_config(self, mock_settings, mock_cvs, mock_sc,
                                     mock_vsi, mock_embed, mock_chroma):
        temp_dir = tempfile.mkdtemp()
        try:
            project_path = os.path.join(temp_dir, "test_project")
            os.makedirs(os.path.join(project_path, "corpus"), exist_ok=True)

            mock_embed.return_value = MagicMock()
            mock_collection = MagicMock()
            mock_client = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chroma.return_value = mock_client

            mock_llm_manager = MagicMock()
            mock_llm_manager.get_llm.return_value = MagicMock()

            config = {
                'embedding_settings': {'model_name': 'test-model'},
                'analysis_settings': {'metadata_keys_to_hide_display': []},
                'ingestion_config': {'cogarc_settings': {}}
            }

            from core.vector_manager import VectorManager
            vm = VectorManager(config, "test_project", project_path, mock_llm_manager)
            self.assertEqual(vm.project_name, "test_project")
            self.assertIsNotNone(vm.embed_model)
        finally:
            shutil.rmtree(temp_dir)


class TestVectorManagerImport(unittest.TestCase):
    """Reports whether vector_manager can be imported."""
    def test_import_status(self):
        if not VECTOR_IMPORT_OK:
            self.skipTest(f"vector_manager not importable: {VECTOR_IMPORT_ERROR}")
        self.assertTrue(VECTOR_IMPORT_OK)


# ── Shared factory ─────────────────────────────────────────────────────────────

def _make_vm(project_path):
    """Return a VectorManager with all heavy deps mocked, rooted at project_path."""
    with patch('core.vector_manager.get_chroma_client') as mock_chroma, \
         patch('core.vector_manager.get_embed_model') as mock_embed, \
         patch('core.vector_manager.VectorStoreIndex'), \
         patch('core.vector_manager.StorageContext'), \
         patch('core.vector_manager.ChromaVectorStore'), \
         patch('core.vector_manager.LlamaSettings'):

        mock_embed.return_value = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = MagicMock()
        mock_chroma.return_value = mock_client
        mock_llm = MagicMock()
        mock_llm.get_llm.return_value = MagicMock()

        config = {
            'embedding_settings': {'model_name': 'test-model'},
            'analysis_settings': {'metadata_keys_to_hide_display': []},
            'ingestion_config': {'cogarc_settings': {}},
        }

        from core.vector_manager import VectorManager
        return VectorManager(config, "test_project", project_path, mock_llm)


# ── _resolve_original_path ─────────────────────────────────────────────────────

@unittest.skipUnless(VECTOR_IMPORT_OK, "vector_manager import failed (torch/chromadb)")
class TestResolveOriginalPath(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        project_path = os.path.join(self.temp_dir, "proj")
        os.makedirs(os.path.join(project_path, "corpus"), exist_ok=True)
        self.vm = _make_vm(project_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_absolute_path_exists(self):
        """Absolute path that exists on disk is returned."""
        f = Path(self.temp_dir) / "exists.txt"
        f.write_text("hello")
        result = self.vm._resolve_original_path(str(f))
        self.assertIsNotNone(result)
        self.assertEqual(result, f)

    def test_absolute_path_missing_returns_none(self):
        """Non-existent absolute path → None."""
        result = self.vm._resolve_original_path(str(Path(self.temp_dir) / "ghost.txt"))
        self.assertIsNone(result)

    def test_relative_path_resolved_via_cwd(self):
        """Relative name resolved via CWD when the file exists there."""
        cwd = Path.cwd()
        rel_name = "_m3_test_resolve_cwd.txt"
        target = cwd / rel_name
        target.write_text("cwd test")
        try:
            result = self.vm._resolve_original_path(rel_name)
            self.assertIsNotNone(result)
        finally:
            target.unlink(missing_ok=True)

    def test_relative_path_resolved_via_project_parent(self):
        """Relative name resolved via project-root parent when file exists there."""
        parent = Path(self.vm.project_path).parent
        rel_name = "_m3_test_resolve_parent.txt"
        target = parent / rel_name
        target.write_text("parent test")
        try:
            # This will be found via strategy 3 (project root parent) IF it isn't
            # already in CWD.  We use a name unlikely to exist in CWD.
            result = self.vm._resolve_original_path(rel_name)
            self.assertIsNotNone(result)
        finally:
            target.unlink(missing_ok=True)

    def test_all_strategies_fail_returns_none(self):
        """When no strategy locates the file, None is returned."""
        result = self.vm._resolve_original_path("__no_such_file_xyz_m3__.txt")
        self.assertIsNone(result)


# ── scan_for_updates ───────────────────────────────────────────────────────────

@unittest.skipUnless(VECTOR_IMPORT_OK, "vector_manager import failed (torch/chromadb)")
class TestScanForUpdates(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = os.path.join(self.temp_dir, "proj")
        os.makedirs(os.path.join(self.project_path, "corpus"), exist_ok=True)
        self.vm = _make_vm(self.project_path)
        self.src_dir = os.path.join(self.temp_dir, "sources")
        os.makedirs(self.src_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _write_metadata(self, entries: dict):
        with open(self.vm.metadata_path, 'w') as f:
            json.dump(entries, f)

    def _src_file(self, name: str, content: str = "content") -> Path:
        p = Path(self.src_dir) / name
        p.write_text(content)
        return p

    def _corpus_key(self, name: str) -> str:
        return os.path.join(self.project_path, "corpus", name)

    def test_empty_corpus_returns_empty(self):
        result = self.vm.scan_for_updates()
        self.assertEqual(result, {'changed': [], 'new': [], 'missing': []})

    def test_unchanged_file_not_reported(self):
        src = self._src_file("doc.txt", "same content")
        from core.vector_manager import get_file_hash
        h = get_file_hash(src)
        corpus_key = self._corpus_key("doc.txt")
        self._write_metadata({corpus_key: {'original_path': str(src), 'hash': h}})

        result = self.vm.scan_for_updates()
        self.assertEqual(result['changed'], [])
        self.assertEqual(result['missing'], [])

    def test_changed_file_reported(self):
        src = self._src_file("doc.txt", "original content")
        old_hash = "0" * 64  # deliberately wrong hash
        corpus_key = self._corpus_key("doc.txt")
        self._write_metadata({corpus_key: {'original_path': str(src), 'hash': old_hash}})

        result = self.vm.scan_for_updates()
        self.assertEqual(len(result['changed']), 1)
        self.assertEqual(result['changed'][0][1], corpus_key)
        self.assertEqual(result['missing'], [])

    def test_missing_file_reported(self):
        ghost = str(Path(self.src_dir) / "gone.txt")  # does not exist
        corpus_key = self._corpus_key("gone.txt")
        self._write_metadata({corpus_key: {'original_path': ghost, 'hash': 'abc'}})

        result = self.vm.scan_for_updates()
        self.assertEqual(len(result['missing']), 1)
        self.assertEqual(result['missing'][0][0], ghost)
        self.assertEqual(result['changed'], [])

    def test_new_file_in_source_dir_reported(self):
        # An existing corpus entry keeps its source dir known
        src_old = self._src_file("old.txt", "old")
        from core.vector_manager import get_file_hash
        h = get_file_hash(src_old)
        corpus_key = self._corpus_key("old.txt")
        self._write_metadata({corpus_key: {'original_path': str(src_old), 'hash': h}})

        # Drop a new file into the same source dir
        new_file = self._src_file("new.txt", "brand new")

        result = self.vm.scan_for_updates()
        self.assertIn(str(new_file.resolve()), result['new'])
        self.assertEqual(result['changed'], [])
        self.assertEqual(result['missing'], [])

    def test_non_scannable_extension_ignored(self):
        src_old = self._src_file("doc.txt", "content")
        from core.vector_manager import get_file_hash
        h = get_file_hash(src_old)
        corpus_key = self._corpus_key("doc.txt")
        self._write_metadata({corpus_key: {'original_path': str(src_old), 'hash': h}})

        # Drop a .csv file (not in _SCANNABLE_EXTENSIONS) into the source dir
        (Path(self.src_dir) / "data.csv").write_text("a,b,c")

        result = self.vm.scan_for_updates()
        self.assertEqual(result['new'], [])


# ── reingest_changed_file ──────────────────────────────────────────────────────

@unittest.skipUnless(VECTOR_IMPORT_OK, "vector_manager import failed (torch/chromadb)")
class TestReingestChangedFile(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = os.path.join(self.temp_dir, "proj")
        os.makedirs(os.path.join(self.project_path, "corpus"), exist_ok=True)
        self.vm = _make_vm(self.project_path)
        self.src_dir = os.path.join(self.temp_dir, "sources")
        os.makedirs(self.src_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _write_metadata(self, entries: dict):
        with open(self.vm.metadata_path, 'w') as f:
            json.dump(entries, f)

    def test_returns_false_when_source_missing(self):
        ghost = str(Path(self.src_dir) / "gone.txt")
        corpus_key = os.path.join(self.project_path, "corpus", "gone.txt")
        meta = {'original_path': ghost, 'hash': 'abc', 'doc_type': 'document'}
        self._write_metadata({corpus_key: meta})

        result = self.vm.reingest_changed_file(corpus_key, meta)
        self.assertFalse(result)

    def test_successful_reingest_returns_true_and_preserves_history(self):
        src = Path(self.src_dir) / "doc.txt"
        src.write_text("updated content")

        corpus_key = os.path.join(self.project_path, "corpus", "old-uuid.txt")
        # Create the corpus binary so unlink() in reingest doesn't fail
        Path(corpus_key).write_text("old content")

        old_hash = "0" * 64
        old_meta = {
            'original_path': str(src),
            'doc_type': 'document',
            'hash': old_hash,
            'added_at': '2024-01-01T00:00:00+00:00',
            'pipeline': 'cogarc',
            'chunk_count': 3,
            'text_cache_path': None,
            'version_history': [],
        }
        self._write_metadata({corpus_key: old_meta})

        # Mock collection interactions and the heavy pipeline call
        self.vm.collection.get.return_value = {'ids': ['c1', 'c2']}

        mock_doc = MagicMock()
        mock_doc.get_content.return_value = "updated content"

        with patch('core.vector_manager.read_files', return_value=[mock_doc]), \
             patch.object(self.vm, '_process_and_ingest_file', return_value=[MagicMock(), MagicMock()]):

            result = self.vm.reingest_changed_file(corpus_key, old_meta)

        self.assertTrue(result)

        # Old chunks should have been deleted
        self.vm.collection.delete.assert_called_once_with(ids=['c1', 'c2'])

        # New metadata entry should exist with version history
        new_meta = self.vm._load_metadata()
        self.assertNotIn(corpus_key, new_meta, "Old corpus key should be removed")
        self.assertEqual(len(new_meta), 1)
        new_entry = list(new_meta.values())[0]
        self.assertEqual(len(new_entry['version_history']), 1)
        self.assertEqual(new_entry['version_history'][0]['old_hash'], old_hash)
        self.assertEqual(new_entry['chunk_count'], 2)  # two mock nodes returned


if __name__ == '__main__':
    unittest.main()
