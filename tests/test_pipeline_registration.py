"""
Unit tests for pipeline stage registration:
  - register_stage(), unregister_stage(), list_stages()
  - _resolve_stage_order() position handling
  - StageRegistration dataclass

NOTE: These tests require llama_index (and transitively torch) to be available.
"""
import unittest

try:
    from unittest.mock import patch, MagicMock
    from llama_index.core.schema import Document, TextNode
    PIPELINE_IMPORT_OK = True
except (ImportError, OSError) as e:
    PIPELINE_IMPORT_OK = False
    PIPELINE_IMPORT_ERROR = str(e)


def _make_pipeline():
    """Helper: create a CognitiveArchitectPipeline with mocked LLM."""
    mock_llm_manager = MagicMock()
    mock_llm_manager.get_llm.return_value = MagicMock()
    config = {
        'ingestion_config': {
            'cogarc_settings': {
                'stage_0_model': 'stratify_model',
                'stage_1_model': 'synthesis_model',
                'stage_2_model': 'enrichment_model',
                'stage_3_model': 'synthesis_model',
            }
        },
        'analysis_settings': {
            'metadata_keys_to_embed': ['themes']
        }
    }
    from core.ingestion.cognitive_architect_pipeline import CognitiveArchitectPipeline
    return CognitiveArchitectPipeline(config, mock_llm_manager)


# ─────────────────────────────────────────────
# Registration API
# ─────────────────────────────────────────────

@unittest.skipUnless(PIPELINE_IMPORT_OK, "llama_index/torch not available")
class TestRegisterStage(unittest.TestCase):

    def test_register_and_list(self):
        """Registering a stage makes it appear in list_stages()."""
        pipeline = _make_pipeline()

        def my_stage(data):
            return data

        pipeline.register_stage("my_stage", my_stage, position="append", description="Test stage")
        stages = pipeline.list_stages()
        names = [s["name"] for s in stages]
        self.assertIn("my_stage", names)

        # Verify it's listed as registered type
        my = next(s for s in stages if s["name"] == "my_stage")
        self.assertEqual(my["type"], "registered")
        self.assertEqual(my["description"], "Test stage")

    def test_builtin_stages_present(self):
        """list_stages() includes all 4 built-in stages."""
        pipeline = _make_pipeline()
        stages = pipeline.list_stages()
        builtin_names = [s["name"] for s in stages if s["type"] == "builtin"]
        self.assertIn("stratify", builtin_names)
        self.assertIn("structure", builtin_names)
        self.assertIn("enrich", builtin_names)
        self.assertIn("synthesize", builtin_names)

    def test_duplicate_raises(self):
        """Registering the same name twice raises ValueError."""
        pipeline = _make_pipeline()
        pipeline.register_stage("dup", lambda d: d)
        with self.assertRaises(ValueError):
            pipeline.register_stage("dup", lambda d: d)

    def test_invalid_position_raises(self):
        """Invalid position format raises ValueError."""
        pipeline = _make_pipeline()
        with self.assertRaises(ValueError):
            pipeline.register_stage("bad", lambda d: d, position="invalid_pos")

    def test_unknown_target_raises(self):
        """Position referencing unknown stage raises ValueError."""
        pipeline = _make_pipeline()
        with self.assertRaises(ValueError):
            pipeline.register_stage("bad", lambda d: d, position="after:nonexistent")

    def test_unregister(self):
        """Unregistering removes stage from list_stages()."""
        pipeline = _make_pipeline()
        pipeline.register_stage("temp", lambda d: d)
        self.assertTrue(any(s["name"] == "temp" for s in pipeline.list_stages()))

        pipeline.unregister_stage("temp")
        self.assertFalse(any(s["name"] == "temp" for s in pipeline.list_stages()))

    def test_unregister_nonexistent_raises(self):
        """Unregistering a name that doesn't exist raises ValueError."""
        pipeline = _make_pipeline()
        with self.assertRaises(ValueError):
            pipeline.unregister_stage("nonexistent")


# ─────────────────────────────────────────────
# Position handling
# ─────────────────────────────────────────────

@unittest.skipUnless(PIPELINE_IMPORT_OK, "llama_index/torch not available")
class TestStagePositioning(unittest.TestCase):

    def test_position_append(self):
        """append places stage after all built-ins."""
        pipeline = _make_pipeline()
        pipeline.register_stage("tail", lambda d: d, position="append")
        stages = pipeline.list_stages()
        names = [s["name"] for s in stages]
        self.assertEqual(names[-1], "tail")

    def test_position_after(self):
        """after:enrich places stage right after enrich."""
        pipeline = _make_pipeline()
        pipeline.register_stage("post_enrich", lambda d: d, position="after:enrich")
        stages = pipeline.list_stages()
        names = [s["name"] for s in stages]
        enrich_idx = names.index("enrich")
        self.assertEqual(names[enrich_idx + 1], "post_enrich")

    def test_position_before(self):
        """before:synthesize places stage right before synthesize."""
        pipeline = _make_pipeline()
        pipeline.register_stage("pre_synth", lambda d: d, position="before:synthesize")
        stages = pipeline.list_stages()
        names = [s["name"] for s in stages]
        synth_idx = names.index("synthesize")
        self.assertEqual(names[synth_idx - 1], "pre_synth")

    def test_position_after_registered(self):
        """Can reference a previously registered stage as position target."""
        pipeline = _make_pipeline()
        pipeline.register_stage("first_custom", lambda d: d, position="append")
        pipeline.register_stage("second_custom", lambda d: d, position="after:first_custom")
        stages = pipeline.list_stages()
        names = [s["name"] for s in stages]
        first_idx = names.index("first_custom")
        self.assertEqual(names[first_idx + 1], "second_custom")


# ─────────────────────────────────────────────
# Stage execution
# ─────────────────────────────────────────────

@unittest.skipUnless(PIPELINE_IMPORT_OK, "llama_index/torch not available")
class TestStageExecution(unittest.TestCase):

    @patch('core.ingestion.cognitive_architect_pipeline.CogArcStage3Synthesis')
    @patch('core.ingestion.cognitive_architect_pipeline.CogArcStage2Enrich')
    @patch('core.ingestion.cognitive_architect_pipeline.CogArcStage1Structure')
    @patch('core.ingestion.cognitive_architect_pipeline.CogArcStage0Stratify')
    def test_registered_stage_is_called(self, mock_s0, mock_s1, mock_s2, mock_s3):
        """A registered stage's function is called during run()."""
        mock_llm_manager = MagicMock()
        mock_llm_manager.get_llm.return_value = MagicMock()
        config = {
            'ingestion_config': {
                'cogarc_settings': {
                    'stage_0_model': 'stratify_model',
                    'stage_1_model': 'synthesis_model',
                    'stage_2_model': 'enrichment_model',
                    'stage_3_model': 'synthesis_model',
                }
            },
            'analysis_settings': {'metadata_keys_to_embed': ['themes']}
        }

        # Setup mocked built-in stages to pass data through
        for mock_stage_cls in [mock_s0, mock_s1, mock_s2, mock_s3]:
            inst = MagicMock()
            inst.process.side_effect = lambda d: d
            mock_stage_cls.return_value = inst

        from core.ingestion.cognitive_architect_pipeline import CognitiveArchitectPipeline
        pipeline = CognitiveArchitectPipeline(config, mock_llm_manager)

        call_log = []

        def my_stage(data):
            call_log.append("called")
            data["custom_key"] = "custom_value"
            return data

        pipeline.register_stage("my_stage", my_stage, position="append")

        doc = Document(text="Some text", metadata={'original_filename': 'test.txt'})
        result = pipeline.run([doc], 'document')

        self.assertEqual(call_log, ["called"])
        self.assertEqual(result.get("custom_key"), "custom_value")

    @patch('core.ingestion.cognitive_architect_pipeline.CogArcStage3Synthesis')
    @patch('core.ingestion.cognitive_architect_pipeline.CogArcStage2Enrich')
    @patch('core.ingestion.cognitive_architect_pipeline.CogArcStage1Structure')
    @patch('core.ingestion.cognitive_architect_pipeline.CogArcStage0Stratify')
    def test_no_registered_stages_unchanged_behavior(self, mock_s0, mock_s1, mock_s2, mock_s3):
        """With no registered stages, pipeline runs identically to before."""
        mock_llm_manager = MagicMock()
        mock_llm_manager.get_llm.return_value = MagicMock()
        config = {
            'ingestion_config': {
                'cogarc_settings': {
                    'stage_0_model': 'stratify_model',
                    'stage_1_model': 'synthesis_model',
                    'stage_2_model': 'enrichment_model',
                    'stage_3_model': 'synthesis_model',
                }
            },
            'analysis_settings': {'metadata_keys_to_embed': ['themes']}
        }

        mock_s1_inst = MagicMock()
        mock_s1_inst.process.return_value = {'documents': []}
        mock_s1.return_value = mock_s1_inst

        mock_s2_inst = MagicMock()
        mock_s2_inst.process.return_value = {'primary_nodes': []}
        mock_s2.return_value = mock_s2_inst

        mock_s3_inst = MagicMock()
        mock_s3_inst.process.return_value = {'primary_nodes': []}
        mock_s3.return_value = mock_s3_inst

        from core.ingestion.cognitive_architect_pipeline import CognitiveArchitectPipeline
        pipeline = CognitiveArchitectPipeline(config, mock_llm_manager)

        doc = Document(text="Regular document", metadata={'original_filename': 'doc.txt'})
        result = pipeline.run([doc], 'document')

        # Stage 0 should not be called (non-interview)
        mock_s0.return_value.process.assert_not_called()
        # Stages 1-3 should be called
        mock_s1_inst.process.assert_called_once()
        mock_s2_inst.process.assert_called_once()
        mock_s3_inst.process.assert_called_once()


class TestPipelineRegistrationImport(unittest.TestCase):
    """Reports whether pipeline registration tests can run."""
    def test_import_status(self):
        if not PIPELINE_IMPORT_OK:
            self.skipTest(f"llama_index not importable: {PIPELINE_IMPORT_ERROR}")
        self.assertTrue(PIPELINE_IMPORT_OK)


if __name__ == '__main__':
    unittest.main()
