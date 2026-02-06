"""
Unit tests for the ingestion pipeline:
  - core/ingestion/base_pipeline.py
  - core/ingestion/pipeline_factory.py
  - core/ingestion/cognitive_architect_pipeline.py
  - core/ingestion/stages/ (all 4 stages)

NOTE: These tests require llama_index (and transitively torch) to be available.
If torch is broken in the environment, tests will be skipped.
"""
import unittest

try:
    from unittest.mock import patch, MagicMock, PropertyMock
    from llama_index.core.schema import Document, TextNode
    PIPELINE_IMPORT_OK = True
except (ImportError, OSError) as e:
    PIPELINE_IMPORT_OK = False
    PIPELINE_IMPORT_ERROR = str(e)


# ─────────────────────────────────────────────
# Pipeline Factory
# ─────────────────────────────────────────────

@unittest.skipUnless(PIPELINE_IMPORT_OK, "llama_index/torch not available")
class TestPipelineFactory(unittest.TestCase):
    @patch('core.ingestion.pipeline_factory.PIPELINES')
    def test_get_pipeline_cogarc(self, mock_pipelines):
        mock_pipeline_cls = MagicMock()
        mock_pipelines.get.return_value = mock_pipeline_cls
        from core.ingestion.pipeline_factory import get_pipeline
        pipeline = get_pipeline('cogarc', {}, MagicMock())
        self.assertIsNotNone(pipeline)
        mock_pipeline_cls.assert_called_once()

    def test_get_pipeline_invalid_name(self):
        from core.ingestion.pipeline_factory import get_pipeline
        with self.assertRaises(ValueError):
            get_pipeline('nonexistent_pipeline', {}, MagicMock())


# ─────────────────────────────────────────────
# Stage 0: Q&A Stratification
# ─────────────────────────────────────────────

@unittest.skipUnless(PIPELINE_IMPORT_OK, "llama_index/torch not available")
class TestStage0Stratify(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.config = {}

    def test_stratify_success(self):
        """Test successful Q&A extraction from interview."""
        from core.ingestion.stages.cogarc_stage_0_stratify import CogArcStage0Stratify

        mock_response = MagicMock()
        mock_response.message.content = '[{"question": "What is your role?", "answer": "I am an engineer."}]'
        self.mock_llm.chat.return_value = mock_response

        stage = CogArcStage0Stratify(self.config, llm=self.mock_llm)

        doc = Document(text="Q: What is your role?\nA: I am an engineer.",
                       metadata={'original_filename': 'interview.txt', 'file_path': '/test/interview.txt'})
        data = {'documents': [doc]}

        result = stage.process(data)
        self.assertIn('documents', result)
        self.assertGreater(len(result['documents']), 0)
        self.assertIn('questions', result)

    def test_stratify_fallback_on_bad_json(self):
        """Test fallback when LLM returns invalid JSON."""
        from core.ingestion.stages.cogarc_stage_0_stratify import CogArcStage0Stratify

        mock_response = MagicMock()
        mock_response.message.content = "This is not JSON at all"
        self.mock_llm.chat.return_value = mock_response

        stage = CogArcStage0Stratify(self.config, llm=self.mock_llm)

        doc = Document(text="Some interview text",
                       metadata={'original_filename': 'test.txt', 'file_path': '/test/test.txt'})
        data = {'documents': [doc]}

        result = stage.process(data)
        # Should fall back and include the original doc
        self.assertEqual(len(result['documents']), 1)

    def test_stratify_empty_documents(self):
        """Test stage with no documents."""
        from core.ingestion.stages.cogarc_stage_0_stratify import CogArcStage0Stratify
        stage = CogArcStage0Stratify(self.config, llm=self.mock_llm)
        data = {'documents': []}
        result = stage.process(data)
        self.assertEqual(len(result['documents']), 0)


# ─────────────────────────────────────────────
# Stage 1: Thematic Scaffolding
# ─────────────────────────────────────────────

@unittest.skipUnless(PIPELINE_IMPORT_OK, "llama_index/torch not available")
class TestStage1Structure(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.config = {}

    def test_structure_adds_themes(self):
        """Test that themes are added to document metadata."""
        from core.ingestion.stages.cogarc_stage_1_structure import CogArcStage1Structure

        mock_response = MagicMock()
        mock_response.message.content = '["Machine Learning", "Data Science", "AI Ethics"]'
        self.mock_llm.chat.return_value = mock_response

        stage = CogArcStage1Structure(self.config, llm=self.mock_llm)

        doc = Document(
            text="This is a long enough document about machine learning and data science. " * 5,
            metadata={'original_filename': 'doc.txt'}
        )
        data = {'documents': [doc]}

        result = stage.process(data)
        self.assertIn('themes', result['documents'][0].metadata)

    def test_structure_skips_short_text(self):
        """Test that very short texts are skipped."""
        from core.ingestion.stages.cogarc_stage_1_structure import CogArcStage1Structure

        stage = CogArcStage1Structure(self.config, llm=self.mock_llm)

        doc = Document(text="Short text.", metadata={'original_filename': 'short.txt'})
        data = {'documents': [doc]}

        result = stage.process(data)
        self.mock_llm.chat.assert_not_called()
        self.assertEqual(len(result['documents']), 1)

    def test_structure_empty_documents(self):
        """Test stage with no documents."""
        from core.ingestion.stages.cogarc_stage_1_structure import CogArcStage1Structure
        stage = CogArcStage1Structure(self.config, llm=self.mock_llm)
        data = {'documents': []}
        result = stage.process(data)
        self.assertEqual(len(result['documents']), 0)


# ─────────────────────────────────────────────
# Stage 2: Micro-Context Enrichment
# ─────────────────────────────────────────────

@unittest.skipUnless(PIPELINE_IMPORT_OK, "llama_index/torch not available")
class TestStage2Enrich(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.config = {}

    def test_enrich_generates_questions(self):
        """Test that hypothetical questions are added."""
        from core.ingestion.stages.cogarc_stage_2_enrich import CogArcStage2Enrich

        mock_response = MagicMock()
        mock_response.message.content = "What are the benefits of machine learning?"
        self.mock_llm.chat.return_value = mock_response

        stage = CogArcStage2Enrich(self.config, llm=self.mock_llm)

        doc = Document(
            text="Machine learning offers many benefits including automation. " * 10,
            metadata={'original_filename': 'doc.txt', 'themes': 'ML, AI'}
        )
        data = {'documents': [doc]}

        result = stage.process(data)
        self.assertIn('primary_nodes', result)
        nodes = result['primary_nodes']
        self.assertGreater(len(nodes), 0)
        has_question = any('hypothetical_question' in n.metadata for n in nodes)
        self.assertTrue(has_question)

    def test_enrich_empty_documents(self):
        """Test stage with no documents."""
        from core.ingestion.stages.cogarc_stage_2_enrich import CogArcStage2Enrich
        stage = CogArcStage2Enrich(self.config, llm=self.mock_llm)
        data = {'documents': []}
        result = stage.process(data)
        self.assertNotIn('primary_nodes', result)


# ─────────────────────────────────────────────
# Stage 3: Holistic Synthesis
# ─────────────────────────────────────────────

@unittest.skipUnless(PIPELINE_IMPORT_OK, "llama_index/torch not available")
class TestStage3Synthesis(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock()
        self.config = {}

    def test_synthesis_adds_summary(self):
        """Test that holistic summary is added to all nodes."""
        from core.ingestion.stages.cogarc_stage_3_synthesis import CogArcStage3Synthesis

        mock_response = MagicMock()
        mock_response.message.content = "This document discusses the impact of AI on society."
        self.mock_llm.chat.return_value = mock_response

        stage = CogArcStage3Synthesis(self.config, llm=self.mock_llm)

        nodes = [
            TextNode(text="Chunk 1 about AI", metadata={}),
            TextNode(text="Chunk 2 about AI", metadata={}),
        ]
        data = {'primary_nodes': nodes}

        result = stage.process(data)
        for node in result['primary_nodes']:
            self.assertIn('holistic_summary', node.metadata)

    def test_synthesis_no_nodes(self):
        """Test stage with no nodes."""
        from core.ingestion.stages.cogarc_stage_3_synthesis import CogArcStage3Synthesis
        stage = CogArcStage3Synthesis(self.config, llm=self.mock_llm)
        data = {'primary_nodes': None}
        result = stage.process(data)
        self.assertIsNone(result.get('primary_nodes'))

    def test_synthesis_handles_llm_error(self):
        """Test graceful handling when LLM fails."""
        from core.ingestion.stages.cogarc_stage_3_synthesis import CogArcStage3Synthesis

        self.mock_llm.chat.side_effect = Exception("LLM unavailable")

        stage = CogArcStage3Synthesis(self.config, llm=self.mock_llm)
        nodes = [TextNode(text="Some text", metadata={})]
        data = {'primary_nodes': nodes}

        result = stage.process(data)
        self.assertIn('primary_nodes', result)
        self.assertNotIn('holistic_summary', result['primary_nodes'][0].metadata)


# ─────────────────────────────────────────────
# Full CogArc Pipeline (integration-level)
# ─────────────────────────────────────────────

@unittest.skipUnless(PIPELINE_IMPORT_OK, "llama_index/torch not available")
class TestCognitiveArchitectPipeline(unittest.TestCase):
    @patch('core.ingestion.cognitive_architect_pipeline.CogArcStage3Synthesis')
    @patch('core.ingestion.cognitive_architect_pipeline.CogArcStage2Enrich')
    @patch('core.ingestion.cognitive_architect_pipeline.CogArcStage1Structure')
    @patch('core.ingestion.cognitive_architect_pipeline.CogArcStage0Stratify')
    def test_pipeline_skips_stage0_for_documents(self, mock_s0, mock_s1, mock_s2, mock_s3):
        """Test that Stage 0 is skipped for non-interview docs."""
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

        doc = Document(text="Regular document text", metadata={'original_filename': 'doc.txt'})
        result = pipeline.run([doc], 'document')

        mock_s0.return_value.process.assert_not_called()

    @patch('core.ingestion.cognitive_architect_pipeline.CogArcStage3Synthesis')
    @patch('core.ingestion.cognitive_architect_pipeline.CogArcStage2Enrich')
    @patch('core.ingestion.cognitive_architect_pipeline.CogArcStage1Structure')
    @patch('core.ingestion.cognitive_architect_pipeline.CogArcStage0Stratify')
    def test_pipeline_runs_stage0_for_interviews(self, mock_s0, mock_s1, mock_s2, mock_s3):
        """Test that Stage 0 IS called for interview docs."""
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

        mock_s0_inst = MagicMock()
        mock_s0_inst.process.return_value = {
            'documents': [Document(text="answer text", metadata={'original_filename': 'i.txt'})]
        }
        mock_s0.return_value = mock_s0_inst

        mock_s1_inst = MagicMock()
        mock_s1_inst.process.return_value = {'documents': [Document(text="text", metadata={})]}
        mock_s1.return_value = mock_s1_inst

        mock_s2_inst = MagicMock()
        mock_s2_inst.process.return_value = {'primary_nodes': [TextNode(text="n", metadata={})]}
        mock_s2.return_value = mock_s2_inst

        mock_s3_inst = MagicMock()
        mock_s3_inst.process.return_value = {'primary_nodes': [TextNode(text="n", metadata={})]}
        mock_s3.return_value = mock_s3_inst

        from core.ingestion.cognitive_architect_pipeline import CognitiveArchitectPipeline
        pipeline = CognitiveArchitectPipeline(config, mock_llm_manager)

        doc = Document(text="Interview transcript", metadata={'original_filename': 'interview.txt'})
        result = pipeline.run([doc], 'interview')

        mock_s0_inst.process.assert_called_once()


class TestIngestionPipelineImport(unittest.TestCase):
    """Reports whether the ingestion pipeline can be imported."""
    def test_import_status(self):
        if not PIPELINE_IMPORT_OK:
            self.skipTest(f"llama_index not importable: {PIPELINE_IMPORT_ERROR}")
        self.assertTrue(PIPELINE_IMPORT_OK)


if __name__ == '__main__':
    unittest.main()
