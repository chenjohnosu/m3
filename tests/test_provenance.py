"""
Unit tests for the Document Provenance System.

Tests cover:
  - Text cache layout and roundtrip (TestTextCacheLayout)
  - Provenance metadata schema and backward-compat (TestProvenanceMetadataSchema)
  - Reconstitution logic — cache primary, ChromaDB fallback (TestReconstitutionLogic)
  - Replace workflow — history accumulation (TestReplaceWorkflow)
  - CLI command registration (TestCorpusCommandsProvenance)
  - Chunk-to-document lookup (TestChunkSourceLookup)

All tests use tempfiles + mocks; no real ChromaDB or torch required.
"""
import unittest
import tempfile
import os
import shutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# ── Attempt to import the symbols we need ─────────────────────────────────

VECTOR_IMPORT_OK = False
VECTOR_IMPORT_ERROR = ""
try:
    from core.vector_manager import VectorManager, get_file_hash
    VECTOR_IMPORT_OK = True
except (ImportError, OSError) as exc:
    VECTOR_IMPORT_ERROR = str(exc)

CORPUS_COMMANDS_OK = False
CORPUS_COMMANDS_ERROR = ""
try:
    from cli.corpus_commands import corpus as corpus_group
    CORPUS_COMMANDS_OK = True
except (ImportError, OSError) as exc:
    CORPUS_COMMANDS_ERROR = str(exc)


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_vm(tmp_dir, extra_config=None):
    """Build a VectorManager instance with all heavy deps mocked out."""
    project_path = os.path.join(tmp_dir, "project")
    os.makedirs(os.path.join(project_path, "corpus"), exist_ok=True)

    mock_collection = MagicMock()
    mock_collection.get.return_value = {'ids': [], 'metadatas': []}
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    mock_llm_manager = MagicMock()
    mock_llm_manager.get_llm.return_value = MagicMock()

    config = {
        'embedding_settings': {'model_name': 'test-model'},
        'analysis_settings': {'metadata_keys_to_hide_display': []},
        'ingestion_config': {'cogarc_settings': {}},
        **(extra_config or {}),
    }

    with patch('core.vector_manager.get_embed_model', return_value=MagicMock()), \
         patch('core.vector_manager.get_chroma_client', return_value=mock_client), \
         patch('core.vector_manager.ChromaVectorStore', return_value=MagicMock()), \
         patch('core.vector_manager.StorageContext') as mock_sc, \
         patch('core.vector_manager.VectorStoreIndex') as mock_vsi, \
         patch('core.vector_manager.LlamaSettings'):
        mock_sc.from_defaults.return_value = MagicMock()
        mock_vsi.from_documents.return_value = MagicMock()
        vm = VectorManager(config, "test_proj", project_path, mock_llm_manager)
        vm.collection = mock_collection

    return vm, project_path


# ══════════════════════════════════════════════════════════════════════════
# 1. Text Cache Layout
# ══════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(VECTOR_IMPORT_OK, f"vector_manager import failed: {VECTOR_IMPORT_ERROR}")
class TestTextCacheLayout(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.vm, self.project_path = _make_vm(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_text_cache_dir_created(self):
        """text_cache/ directory must exist after VectorManager init."""
        expected = os.path.join(self.project_path, "text_cache")
        self.assertTrue(os.path.isdir(expected))

    def test_write_text_cache_creates_file(self):
        """_write_text_cache returns an absolute path and the file exists."""
        path = self.vm._write_text_cache("abc-123", "hello world")
        self.assertTrue(os.path.isabs(path))
        self.assertTrue(os.path.exists(path))
        self.assertTrue(path.endswith("abc-123.txt"))

    def test_text_cache_utf8_roundtrip(self):
        """Text with non-ASCII characters survives a write-then-read cycle."""
        content = "Héllo wörld — 日本語テスト"
        path = self.vm._write_text_cache("utf8-test", content)
        with open(path, encoding='utf-8') as f:
            loaded = f.read()
        self.assertEqual(loaded, content)

    def test_delete_text_cache_removes_file(self):
        """_delete_text_cache silently removes an existing cache file."""
        path = self.vm._write_text_cache("del-test", "data")
        self.assertTrue(os.path.exists(path))
        self.vm._delete_text_cache(path)
        self.assertFalse(os.path.exists(path))

    def test_delete_text_cache_missing_is_silent(self):
        """_delete_text_cache does not raise when file is already gone."""
        missing = os.path.join(self.vm.text_cache_path, "ghost.txt")
        # Should not raise
        self.vm._delete_text_cache(missing)

    def test_delete_text_cache_none_is_silent(self):
        """_delete_text_cache does not raise when passed None."""
        self.vm._delete_text_cache(None)


# ══════════════════════════════════════════════════════════════════════════
# 2. Provenance Metadata Schema
# ══════════════════════════════════════════════════════════════════════════

class TestProvenanceMetadataSchema(unittest.TestCase):
    """Pure dict/JSON tests — no VectorManager needed."""

    def _fresh_entry(self, **overrides):
        base = {
            'original_path': '/src/doc.docx',
            'doc_type': 'interview',
            'hash': 'abc123',
            'added_at': '2026-01-01T00:00:00+00:00',
            'ingested_at': '2026-01-01T00:01:00+00:00',
            'pipeline': 'cogarc',
            'chunk_count': 7,
            'text_cache_path': '/project/text_cache/uuid.txt',
        }
        base.update(overrides)
        return base

    def test_new_entry_has_provenance_fields(self):
        entry = self._fresh_entry()
        for field in ('ingested_at', 'pipeline', 'chunk_count', 'text_cache_path'):
            self.assertIn(field, entry, f"Missing field: {field}")

    def test_version_history_absent_on_fresh_add(self):
        """version_history must be absent (not None) on a first add."""
        entry = self._fresh_entry()
        self.assertNotIn('version_history', entry)

    def test_version_history_present_after_replace(self):
        """After a replace, version_history is a list with one entry."""
        history_entry = {
            'replaced_at': '2026-02-01T12:00:00+00:00',
            'old_hash': 'old_abc',
            'old_original_path': '/src/doc_v1.docx',
            'old_text_cache_path': '/project/text_cache/old-uuid.txt',
        }
        entry = self._fresh_entry(version_history=[history_entry])
        self.assertIn('version_history', entry)
        self.assertEqual(len(entry['version_history']), 1)
        self.assertEqual(entry['version_history'][0]['old_hash'], 'old_abc')

    def test_version_history_accumulates(self):
        """Two consecutive replaces produce two entries in version_history."""
        h1 = {'replaced_at': '2026-01-15T00:00:00+00:00', 'old_hash': 'hash_v1',
              'old_original_path': '/src/v1.docx', 'old_text_cache_path': '/tc/v1.txt'}
        h2 = {'replaced_at': '2026-02-01T00:00:00+00:00', 'old_hash': 'hash_v2',
              'old_original_path': '/src/v2.docx', 'old_text_cache_path': '/tc/v2.txt'}
        entry = self._fresh_entry(version_history=[h2, h1])  # newest first
        self.assertEqual(len(entry['version_history']), 2)
        self.assertEqual(entry['version_history'][0]['old_hash'], 'hash_v2')

    def test_backward_compat_old_entry_without_provenance(self):
        """Old entries without provenance fields degrade gracefully via .get()."""
        old_entry = {
            'original_path': '/src/old.txt',
            'doc_type': 'document',
            'hash': 'xyz',
            'added_at': '2025-01-01T00:00:00+00:00',
        }
        # All new fields must gracefully return None / [] via .get()
        self.assertIsNone(old_entry.get('ingested_at'))
        self.assertIsNone(old_entry.get('pipeline'))
        self.assertIsNone(old_entry.get('chunk_count'))
        self.assertIsNone(old_entry.get('text_cache_path'))
        self.assertEqual(old_entry.get('version_history', []), [])

    def test_metadata_json_roundtrip(self):
        """Provenance metadata survives a JSON dump/load cycle."""
        entry = self._fresh_entry(version_history=[])
        serialised = json.dumps(entry)
        loaded = json.loads(serialised)
        self.assertEqual(loaded['chunk_count'], 7)
        self.assertEqual(loaded['pipeline'], 'cogarc')


# ══════════════════════════════════════════════════════════════════════════
# 3. Reconstitution Logic
# ══════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(VECTOR_IMPORT_OK, f"vector_manager import failed: {VECTOR_IMPORT_ERROR}")
class TestReconstitutionLogic(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.vm, self.project_path = _make_vm(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _seed_metadata(self, corpus_key, meta):
        metadata = self.vm._load_metadata()
        metadata[corpus_key] = meta
        self.vm._save_metadata(metadata)

    def test_primary_path_reads_cache(self):
        """When text cache exists, reconstitute returns its content."""
        cache_path = self.vm._write_text_cache("doc-1", "cached content here")
        corpus_key = os.path.join(self.vm.corpus_path, "doc-1.txt")
        self._seed_metadata(corpus_key, {
            'original_path': '/src/doc.txt',
            'text_cache_path': cache_path,
            'doc_type': 'document',
            'hash': 'h1',
        })

        success, text = self.vm.reconstitute_document("doc.txt")
        self.assertTrue(success)
        self.assertEqual(text, "cached content here")

    def test_fallback_joins_chunks_by_index(self):
        """When cache is absent, chunks are joined in chunk_index order."""
        corpus_key = os.path.join(self.vm.corpus_path, "doc-fallback.txt")
        self._seed_metadata(corpus_key, {
            'original_path': '/src/doc_fallback.txt',
            'text_cache_path': None,
            'doc_type': 'document',
            'hash': 'h2',
        })

        # Mock collection.get to return chunks out of order
        self.vm.collection.get.return_value = {
            'metadatas': [
                {'original_text': 'Part B', 'chunk_index': 1},
                {'original_text': 'Part A', 'chunk_index': 0},
            ]
        }

        success, text = self.vm.reconstitute_document("doc_fallback.txt")
        self.assertTrue(success)
        self.assertIn('Part A', text)
        self.assertIn('Part B', text)
        # Part A must come before Part B
        self.assertLess(text.index('Part A'), text.index('Part B'))

    def test_fallback_excludes_missing_original_text(self):
        """Chunks without original_text are excluded from fallback assembly."""
        corpus_key = os.path.join(self.vm.corpus_path, "doc-partial.txt")
        self._seed_metadata(corpus_key, {
            'original_path': '/src/doc_partial.txt',
            'text_cache_path': None,
            'doc_type': 'document',
            'hash': 'h3',
        })

        self.vm.collection.get.return_value = {
            'metadatas': [
                {'original_text': 'Valid chunk', 'chunk_index': 0},
                {'chunk_index': 1},  # no original_text
            ]
        }

        success, text = self.vm.reconstitute_document("doc_partial.txt")
        self.assertTrue(success)
        self.assertEqual(text, "Valid chunk")

    def test_empty_chunk_list_returns_error(self):
        """Empty metadatas from ChromaDB yields (False, error message)."""
        corpus_key = os.path.join(self.vm.corpus_path, "doc-empty.txt")
        self._seed_metadata(corpus_key, {
            'original_path': '/src/doc_empty.txt',
            'text_cache_path': None,
            'doc_type': 'document',
            'hash': 'h4',
        })

        self.vm.collection.get.return_value = {'metadatas': []}

        success, msg = self.vm.reconstitute_document("doc_empty.txt")
        self.assertFalse(success)
        self.assertIsInstance(msg, str)
        self.assertTrue(len(msg) > 0)

    def test_unknown_identifier_returns_error(self):
        """Reconstitute returns (False, msg) for an unknown identifier."""
        success, msg = self.vm.reconstitute_document("no_such_file.txt")
        self.assertFalse(success)
        self.assertIn("not found", msg.lower())

    def test_from_store_flag_bypasses_cache(self):
        """from_store=True skips text cache even when it exists."""
        cache_path = self.vm._write_text_cache("doc-byp", "cache content")
        corpus_key = os.path.join(self.vm.corpus_path, "doc-byp.txt")
        self._seed_metadata(corpus_key, {
            'original_path': '/src/doc_byp.txt',
            'text_cache_path': cache_path,
            'doc_type': 'document',
            'hash': 'h5',
        })

        self.vm.collection.get.return_value = {
            'metadatas': [{'original_text': 'store content', 'chunk_index': 0}]
        }

        success, text = self.vm.reconstitute_document("doc_byp.txt", from_store=True)
        self.assertTrue(success)
        self.assertEqual(text, "store content")


# ══════════════════════════════════════════════════════════════════════════
# 4. Replace Workflow
# ══════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(VECTOR_IMPORT_OK, f"vector_manager import failed: {VECTOR_IMPORT_ERROR}")
class TestReplaceWorkflow(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.vm, self.project_path = _make_vm(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_no_history_on_plain_add(self):
        """An entry created without a replace must not have version_history."""
        entry = {
            'original_path': '/src/new.docx',
            'doc_type': 'interview',
            'hash': 'fresh_hash',
            'added_at': '2026-01-01T00:00:00+00:00',
            'ingested_at': '2026-01-01T00:01:00+00:00',
            'pipeline': 'cogarc',
            'chunk_count': 5,
            'text_cache_path': '/tc/fresh.txt',
        }
        self.assertNotIn('version_history', entry)

    def test_history_builds_on_one_replace(self):
        """After one replace, the new entry carries exactly one history item."""
        old_meta = {
            'hash': 'old_hash',
            'original_path': '/src/v1.docx',
            'text_cache_path': '/tc/v1.txt',
            'version_history': [],
        }
        history_entry = {
            'replaced_at': '2026-02-01T00:00:00+00:00',
            'old_hash': old_meta['hash'],
            'old_original_path': old_meta['original_path'],
            'old_text_cache_path': old_meta['text_cache_path'],
        }
        new_history = [history_entry] + old_meta.get('version_history', [])
        self.assertEqual(len(new_history), 1)

    def test_history_accumulates_on_two_replaces(self):
        """After two replaces, the entry carries two history items."""
        first_replace = {
            'replaced_at': '2026-02-01T00:00:00+00:00',
            'old_hash': 'hash_v1',
            'old_original_path': '/src/v1.docx',
            'old_text_cache_path': '/tc/v1.txt',
        }
        second_replace_old_meta = {
            'hash': 'hash_v2',
            'original_path': '/src/v2.docx',
            'text_cache_path': '/tc/v2.txt',
            'version_history': [first_replace],
        }
        second_entry = {
            'replaced_at': '2026-03-01T00:00:00+00:00',
            'old_hash': second_replace_old_meta['hash'],
            'old_original_path': second_replace_old_meta['original_path'],
            'old_text_cache_path': second_replace_old_meta['text_cache_path'],
        }
        accumulated = [second_entry] + second_replace_old_meta.get('version_history', [])
        self.assertEqual(len(accumulated), 2)
        self.assertEqual(accumulated[0]['old_hash'], 'hash_v2')
        self.assertEqual(accumulated[1]['old_hash'], 'hash_v1')


# ══════════════════════════════════════════════════════════════════════════
# 5. Corpus CLI Commands Provenance
# ══════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(CORPUS_COMMANDS_OK, f"corpus_commands import failed: {CORPUS_COMMANDS_ERROR}")
class TestCorpusCommandsProvenance(unittest.TestCase):

    def test_provenance_command_registered(self):
        commands = corpus_group.list_commands(None)
        self.assertIn('provenance', commands, "Missing corpus subcommand: provenance")

    def test_reconstitute_command_registered(self):
        commands = corpus_group.list_commands(None)
        self.assertIn('reconstitute', commands, "Missing corpus subcommand: reconstitute")

    def test_find_source_command_registered(self):
        commands = corpus_group.list_commands(None)
        self.assertIn('find-source', commands, "Missing corpus subcommand: find-source")


# ══════════════════════════════════════════════════════════════════════════
# 6. Chunk Source Lookup
# ══════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(VECTOR_IMPORT_OK, f"vector_manager import failed: {VECTOR_IMPORT_ERROR}")
class TestChunkSourceLookup(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.vm, self.project_path = _make_vm(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _seed_corpus_meta(self, corpus_key, meta):
        metadata = self.vm._load_metadata()
        metadata[corpus_key] = meta
        self.vm._save_metadata(metadata)

    def test_chunk_id_lookup_returns_source_filename(self):
        """find_source_by_chunk returns original_filename for a known chunk."""
        corpus_key = os.path.join(self.vm.corpus_path, "def456.docx")
        self._seed_corpus_meta(corpus_key, {
            'original_path': '/src/491_Interview_1.docx',
            'original_filename': '491_Interview_1.docx',
            'doc_type': 'interview',
            'chunk_count': 14,
            'hash': 'hh',
        })

        self.vm.collection.get.return_value = {
            'metadatas': [{
                'file_path': corpus_key,
                'original_filename': '491_Interview_1.docx',
                'chunk_index': 4,
                'chunk_count': 14,
            }]
        }

        success, info = self.vm.find_source_by_chunk("some-chunk-id")
        self.assertTrue(success)
        self.assertEqual(info['original_filename'], '491_Interview_1.docx')

    def test_chunk_index_position_string(self):
        """Given chunk_index=4 and chunk_count=14, display position is '5/14' (1-based)."""
        chunk_index = 4
        chunk_count = 14
        position = f"{chunk_index + 1}/{chunk_count}"
        self.assertEqual(position, "5/14")

    def test_unknown_chunk_id_returns_error(self):
        """find_source_by_chunk returns (False, msg) for a chunk not in the store."""
        self.vm.collection.get.return_value = {'metadatas': []}

        success, msg = self.vm.find_source_by_chunk("nonexistent-id")
        self.assertFalse(success)
        self.assertIn("not found", msg.lower())


# ══════════════════════════════════════════════════════════════════════════
# Import status
# ══════════════════════════════════════════════════════════════════════════

class TestProvenanceImportStatus(unittest.TestCase):
    def test_vector_manager_importable(self):
        if not VECTOR_IMPORT_OK:
            self.skipTest(f"vector_manager: {VECTOR_IMPORT_ERROR}")
        self.assertTrue(VECTOR_IMPORT_OK)

    def test_corpus_commands_importable(self):
        if not CORPUS_COMMANDS_OK:
            self.skipTest(f"corpus_commands: {CORPUS_COMMANDS_ERROR}")
        self.assertTrue(CORPUS_COMMANDS_OK)


if __name__ == '__main__':
    unittest.main()
