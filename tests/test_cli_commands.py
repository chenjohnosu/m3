"""
Unit tests for CLI command groups:
  - cli/project_commands.py
  - cli/corpus_commands.py
  - cli/vector_commands.py
  - cli/analyze_commands.py

NOTE: Some CLI modules import chromadb (via VectorManager/AnalyzeManager).
If torch is broken, those imports will fail and tests will be skipped.
"""
import unittest
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
import utils.config as config_module

# Check which CLI modules can be imported
CLI_IMPORT_ERRORS = {}

try:
    from cli.project_commands import project as project_group
    PROJECT_OK = True
except (ImportError, OSError) as e:
    PROJECT_OK = False
    CLI_IMPORT_ERRORS['project'] = str(e)

try:
    from cli.analyze_commands import analyze as analyze_group
    ANALYZE_OK = True
except (ImportError, OSError) as e:
    ANALYZE_OK = False
    CLI_IMPORT_ERRORS['analyze'] = str(e)

try:
    from cli.vector_commands import vector as vector_group
    VECTOR_OK = True
except (ImportError, OSError) as e:
    VECTOR_OK = False
    CLI_IMPORT_ERRORS['vector'] = str(e)

try:
    from cli.corpus_commands import corpus as corpus_group
    CORPUS_OK = True
except (ImportError, OSError) as e:
    CORPUS_OK = False
    CLI_IMPORT_ERRORS['corpus'] = str(e)

try:
    from m3 import cli as main_cli
    MAIN_CLI_OK = True
except (ImportError, OSError) as e:
    MAIN_CLI_OK = False
    CLI_IMPORT_ERRORS['m3'] = str(e)


# ─────────────────────────────────────────────
# Project Commands
# ─────────────────────────────────────────────

@unittest.skipUnless(PROJECT_OK, "project_commands import failed")
class TestProjectCommands(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.projects_dir = os.path.join(self.temp_dir, "projects")
        os.makedirs(self.projects_dir, exist_ok=True)

        self.fake_config = {
            'project_settings': {'projects_directory': self.projects_dir},
            'llm_providers': {},
            'ingestion_config': {
                'known_doc_types': ['document', 'interview'],
                'default_doc_type': 'document',
                'cogarc_settings': {}
            }
        }
        self._original_config = config_module._config
        config_module._config = self.fake_config

        self.runner = CliRunner()

    def tearDown(self):
        config_module._config = self._original_config
        shutil.rmtree(self.temp_dir)

    def test_project_create(self):
        result = self.runner.invoke(project_group, ['create', 'test_proj'], obj=None)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Success", result.output)

    def test_project_create_duplicate(self):
        self.runner.invoke(project_group, ['create', 'dup_proj'], obj=None)
        result = self.runner.invoke(project_group, ['create', 'dup_proj'], obj=None)
        self.assertIn("Error", result.output)

    def test_project_list_empty(self):
        result = self.runner.invoke(project_group, ['list'], obj=None)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No projects found", result.output)

    def test_project_list_with_projects(self):
        self.runner.invoke(project_group, ['create', 'proj_a'], obj=None)
        result = self.runner.invoke(project_group, ['list'], obj=None)
        self.assertIn("proj_a", result.output)

    def test_project_active(self):
        self.runner.invoke(project_group, ['create', 'act_proj'], obj=None)
        result = self.runner.invoke(project_group, ['active', 'act_proj'], obj=None)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Success", result.output)

    def test_project_active_nonexistent(self):
        result = self.runner.invoke(project_group, ['active', 'ghost'], obj=None)
        self.assertIn("Error", result.output)


# ─────────────────────────────────────────────
# Analyze Commands (structure tests)
# ─────────────────────────────────────────────

@unittest.skipUnless(ANALYZE_OK, "analyze_commands import failed")
class TestAnalyzeCommandsStructure(unittest.TestCase):
    """Test that analyze command group has expected subcommands."""

    def test_analyze_group_has_subcommands(self):
        commands = analyze_group.list_commands(None)
        expected = ['topk', 'search', 'exact', 'tools', 'run']
        for cmd in expected:
            self.assertIn(cmd, commands, f"Missing analyze subcommand: {cmd}")


# ─────────────────────────────────────────────
# Vector Commands (structure tests)
# ─────────────────────────────────────────────

@unittest.skipUnless(VECTOR_OK, "vector_commands import failed")
class TestVectorCommandsStructure(unittest.TestCase):
    """Test that vector command group has expected subcommands."""

    def test_vector_group_has_subcommands(self):
        commands = vector_group.list_commands(None)
        expected = ['ingest', 'chunks', 'status', 'rebuild', 'create', 'query']
        for cmd in expected:
            self.assertIn(cmd, commands, f"Missing vector subcommand: {cmd}")


# ─────────────────────────────────────────────
# Corpus Commands (structure tests)
# ─────────────────────────────────────────────

@unittest.skipUnless(CORPUS_OK, "corpus_commands import failed")
class TestCorpusCommandsStructure(unittest.TestCase):
    """Test that corpus command group has expected subcommands."""

    def test_corpus_group_has_subcommands(self):
        commands = corpus_group.list_commands(None)
        expected = ['add', 'remove', 'list', 'ingest', 'rebuild', 'summary',
                    'provenance', 'reconstitute', 'find-source']
        for cmd in expected:
            self.assertIn(cmd, commands, f"Missing corpus subcommand: {cmd}")


# ─────────────────────────────────────────────
# Main CLI entry point
# ─────────────────────────────────────────────

@unittest.skipUnless(MAIN_CLI_OK, "m3 CLI import failed")
class TestMainCLI(unittest.TestCase):
    def test_cli_has_command_groups(self):
        """Verify the main CLI registers all command groups."""
        commands = main_cli.list_commands(None)
        expected = ['project', 'corpus', 'vector', 'analyze']
        for cmd in expected:
            self.assertIn(cmd, commands, f"Missing CLI command group: {cmd}")

    def test_cli_help_output(self):
        """Verify the CLI shows help without error."""
        runner = CliRunner()
        result = runner.invoke(main_cli, ['--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("m3", result.output)


# ─────────────────────────────────────────────
# Import status report
# ─────────────────────────────────────────────

class TestCLIImportStatus(unittest.TestCase):
    def test_project_commands_importable(self):
        if not PROJECT_OK:
            self.skipTest(f"project_commands: {CLI_IMPORT_ERRORS.get('project', 'unknown')}")
        self.assertTrue(PROJECT_OK)

    def test_analyze_commands_importable(self):
        if not ANALYZE_OK:
            self.skipTest(f"analyze_commands: {CLI_IMPORT_ERRORS.get('analyze', 'unknown')}")
        self.assertTrue(ANALYZE_OK)

    def test_vector_commands_importable(self):
        if not VECTOR_OK:
            self.skipTest(f"vector_commands: {CLI_IMPORT_ERRORS.get('vector', 'unknown')}")
        self.assertTrue(VECTOR_OK)

    def test_corpus_commands_importable(self):
        if not CORPUS_OK:
            self.skipTest(f"corpus_commands: {CLI_IMPORT_ERRORS.get('corpus', 'unknown')}")
        self.assertTrue(CORPUS_OK)

    def test_main_cli_importable(self):
        if not MAIN_CLI_OK:
            self.skipTest(f"m3 CLI: {CLI_IMPORT_ERRORS.get('m3', 'unknown')}")
        self.assertTrue(MAIN_CLI_OK)


if __name__ == '__main__':
    unittest.main()
