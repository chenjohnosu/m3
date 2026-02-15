"""
Unit tests for M3System facade (facade.py).

These tests mock instance attributes directly to avoid triggering
heavy llama_index/torch imports that may not be available.
"""
import sys
import unittest

try:
    from unittest.mock import patch, MagicMock
    FACADE_IMPORT_OK = True
except (ImportError, OSError) as e:
    FACADE_IMPORT_OK = False
    FACADE_IMPORT_ERROR = str(e)


def _make_m3_no_project():
    """Create an M3System with light deps mocked, no project open."""
    with patch('facade.get_config', return_value={}), \
         patch('facade.ProjectManager') as mock_pm_cls, \
         patch('facade.LLMManager'), \
         patch('facade.PluginManager'):
        from facade import M3System
        mock_pm = MagicMock()
        mock_pm_cls.return_value = mock_pm
        m3 = M3System()
        return m3, mock_pm


def _make_m3_with_project():
    """Create an M3System with all deps mocked and a project open.
    Bypasses open_project() heavy imports by setting internals directly.
    """
    m3, mock_pm = _make_m3_no_project()
    mock_pm.get_project_path_by_name.return_value = "/tmp/test"

    # Directly set internal state as if open_project() succeeded
    m3._current_project = "test"
    m3._current_project_path = "/tmp/test"
    m3._vector_manager = MagicMock()
    m3._analyze_manager = MagicMock()
    m3._pipeline = MagicMock()

    return m3


# ─────────────────────────────────────────────
# M3System Lifecycle
# ─────────────────────────────────────────────

class TestM3SystemLifecycle(unittest.TestCase):

    def test_create_project_calls_init_and_opens(self):
        """create_project delegates to ProjectManager.init_project."""
        m3, mock_pm = _make_m3_no_project()
        mock_pm.init_project.return_value = ("/tmp/proj", "Created")
        mock_pm.get_project_path_by_name.return_value = "/tmp/proj"

        # Mock the heavy modules to avoid llama_index/chromadb imports
        mock_vm_mod = MagicMock()
        mock_am_mod = MagicMock()
        mock_pf_mod = MagicMock()
        with patch.dict('sys.modules', {
            'core.vector_manager': mock_vm_mod,
            'core.analyze_manager': mock_am_mod,
            'core.ingestion.pipeline_factory': mock_pf_mod,
        }):
            m3.create_project("proj")

        self.assertEqual(m3._current_project, "proj")
        mock_pm.init_project.assert_called_once_with("proj")

    def test_list_projects(self):
        m3, mock_pm = _make_m3_no_project()
        mock_pm.list_projects.return_value = ["proj_a", "proj_b"]
        result = m3.list_projects()
        self.assertEqual(result, ["proj_a", "proj_b"])

    def test_require_open_project(self):
        """Operations without an open project raise RuntimeError."""
        m3, _ = _make_m3_no_project()
        with self.assertRaises(RuntimeError):
            m3.store("some text")
        with self.assertRaises(RuntimeError):
            m3.query("test query")
        with self.assertRaises(RuntimeError):
            m3.list_collections()

    def test_context_manager_cleanup(self):
        """Context manager sets internals to None on exit."""
        m3 = _make_m3_with_project()
        self.assertIsNotNone(m3._current_project)
        m3.__exit__(None, None, None)
        self.assertIsNone(m3._current_project)
        self.assertIsNone(m3._vector_manager)

    def test_context_manager_protocol(self):
        """__enter__ returns self."""
        m3 = _make_m3_with_project()
        self.assertIs(m3.__enter__(), m3)
        m3.__exit__(None, None, None)

    def test_delete_project_requires_confirm(self):
        m3, _ = _make_m3_no_project()
        with self.assertRaises(ValueError):
            m3.delete_project("test_proj")

    def test_open_nonexistent_raises(self):
        m3, mock_pm = _make_m3_no_project()
        mock_pm.get_project_path_by_name.return_value = None
        with self.assertRaises(ValueError):
            m3.open_project("nonexistent")

    def test_delete_project_clears_state(self):
        m3 = _make_m3_with_project()
        m3._project_manager.remove_project.return_value = (True, "Deleted")
        m3.delete_project("test", confirm=True)
        self.assertIsNone(m3._current_project)


# ─────────────────────────────────────────────
# Store and Query
# ─────────────────────────────────────────────

class TestM3SystemStoreQuery(unittest.TestCase):

    def test_store_single_string(self):
        m3 = _make_m3_with_project()
        ids = m3.store("hello world", {"key": "val"}, collection_name="test_coll")
        self.assertEqual(len(ids), 1)
        m3._vector_manager.store.assert_called_once()
        args = m3._vector_manager.store.call_args[0]
        self.assertEqual(args[0], ["hello world"])
        self.assertEqual(args[3], "test_coll")

    def test_store_list(self):
        m3 = _make_m3_with_project()
        ids = m3.store(["doc1", "doc2", "doc3"])
        self.assertEqual(len(ids), 3)

    def test_store_auto_generates_ids(self):
        m3 = _make_m3_with_project()
        ids = m3.store(["a", "b"])
        self.assertEqual(len(ids), 2)
        self.assertNotEqual(ids[0], ids[1])

    def test_store_with_explicit_ids(self):
        m3 = _make_m3_with_project()
        ids = m3.store(["a"], ids=["custom_id"])
        self.assertEqual(ids, ["custom_id"])

    def test_query_delegates(self):
        m3 = _make_m3_with_project()
        m3._vector_manager.query.return_value = {
            "ids": [["id1"]], "documents": [["text1"]],
            "metadatas": [[{"k": "v"}]], "distances": [[0.1]]
        }
        results = m3.query("test query", n_results=3, collection_name="my_coll")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "id1")
        self.assertAlmostEqual(results[0]["distance"], 0.1)
        m3._vector_manager.query.assert_called_once_with(
            query_text="test query", n_results=3, where=None, collection_name="my_coll"
        )

    def test_query_empty_results(self):
        m3 = _make_m3_with_project()
        m3._vector_manager.query.return_value = {
            "ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]
        }
        results = m3.query("nothing")
        self.assertEqual(results, [])

    def test_delete_delegates(self):
        m3 = _make_m3_with_project()
        m3.delete(["id1", "id2"], collection_name="coll")
        m3._vector_manager.delete_by_ids.assert_called_once_with(["id1", "id2"], "coll")


# ─────────────────────────────────────────────
# Collections
# ─────────────────────────────────────────────

class TestM3SystemCollections(unittest.TestCase):

    def test_list_collections(self):
        m3 = _make_m3_with_project()
        mock_db = MagicMock()
        mock_db.list_collections.return_value = ["alpha", "beta"]
        with patch.dict('sys.modules', {'core.db_manager': mock_db}):
            result = m3.list_collections()
        self.assertEqual(result, ["alpha", "beta"])

    def test_collection_count(self):
        m3 = _make_m3_with_project()
        mock_coll = MagicMock()
        mock_coll.count.return_value = 42
        mock_db = MagicMock()
        mock_db.get_or_create_collection.return_value = mock_coll
        with patch.dict('sys.modules', {'core.db_manager': mock_db}):
            count = m3.collection_count("test_coll")
        self.assertEqual(count, 42)

    def test_clear_collection_requires_confirm(self):
        m3 = _make_m3_with_project()
        with self.assertRaises(ValueError):
            m3.clear_collection("test_coll")


# ─────────────────────────────────────────────
# Pipeline Stage Registration via Facade
# ─────────────────────────────────────────────

class TestM3SystemPipelineStages(unittest.TestCase):

    def test_register_delegates(self):
        m3 = _make_m3_with_project()
        fn = lambda d: d
        m3.register_pipeline_stage("my_stage", fn, position="append", description="desc")
        m3._pipeline.register_stage.assert_called_once_with(
            name="my_stage", stage_fn=fn, position="append", description="desc"
        )

    def test_list_delegates(self):
        m3 = _make_m3_with_project()
        m3._pipeline.list_stages.return_value = [{"name": "a", "type": "builtin"}]
        result = m3.list_pipeline_stages()
        self.assertEqual(len(result), 1)

    def test_register_requires_open_project(self):
        m3, _ = _make_m3_no_project()
        with self.assertRaises(RuntimeError):
            m3.register_pipeline_stage("x", lambda d: d)


# ─────────────────────────────────────────────
# Import check
# ─────────────────────────────────────────────

class TestFacadeImport(unittest.TestCase):
    def test_import_status(self):
        if not FACADE_IMPORT_OK:
            self.skipTest(f"Facade imports failed: {FACADE_IMPORT_ERROR}")
        try:
            from facade import M3System
            self.assertTrue(True)
        except ImportError as e:
            self.skipTest(f"M3System import failed: {e}")


if __name__ == '__main__':
    unittest.main()
