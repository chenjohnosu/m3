import click
from core.ingestion.base_pipeline import BasePipeline
from core.ingestion.stages.cogarc_stage_0_stratify import CogArcStage0Stratify
from core.ingestion.stages.cogarc_stage_1_structure import CogArcStage1Structure
from core.ingestion.stages.cogarc_stage_2_enrich import CogArcStage2Enrich
from core.ingestion.stages.cogarc_stage_3_synthesis import CogArcStage3Synthesis
from core.llm_manager import LLMManager
from llama_index.core.schema import TextNode
import hashlib


class CognitiveArchitectPipeline(BasePipeline):
    def __init__(self, config, llm_manager: LLMManager):
        """
        Initializes the pipeline with a pre-loaded LLMManager.
        """
        # Pass the llm_manager to the super class
        super().__init__(config, llm_manager)
        # Use err=True to match other system-level startup logs
        click.echo("Initializing Cognitive Architect Pipeline...", err=True)

        self.cogarc_settings = config.get('ingestion_config', {}).get('cogarc_settings', {})

        # Load embeddable keys from config
        analysis_config = self.config.get('analysis_settings', {})
        self.embeddable_keys = analysis_config.get('metadata_keys_to_embed', ['themes'])  # Default
        click.echo(f"  > CogArc Pipeline: Embedding metadata keys = {self.embeddable_keys}", err=True)

        # --- MODIFIED: Use self.llm_manager (passed in) ---
        # The get_llm() method will now return cached models
        self.stage_0 = CogArcStage0Stratify(
            config, llm=self.llm_manager.get_llm(self.cogarc_settings['stage_0_model'])
        )
        self.stage_1 = CogArcStage1Structure(
            config, llm=self.llm_manager.get_llm(self.cogarc_settings['stage_1_model'])
        )
        self.stage_2 = CogArcStage2Enrich(
            config, llm=self.llm_manager.get_llm(self.cogarc_settings['stage_2_model'])
        )
        self.stage_3 = CogArcStage3Synthesis(
            config, llm=self.llm_manager.get_llm(self.cogarc_settings['stage_3_model'])
        )
        # --- END MODIFICATION ---

        self.current_file_metadata = {}

    def run(self, documents, doc_type):
        """
        Runs the full ingestion pipeline on a single document.
        """
        print(f"\n--- Starting Cognitive Architect Pipeline for doc_type: '{doc_type}' ---")

        # Store file-level metadata (like original_filename)
        if documents:
            self.current_file_metadata = documents[0].metadata.copy()
            self.current_file_metadata.pop('text', None)  # Remove text, keep metadata

        pipeline_data = {'documents': documents}

        if doc_type == 'interview':
            pipeline_data = self.stage_0.process(pipeline_data)
        else:
            print("Skipping Stage 0 (Q&A Stratification) for non-interview document.")
            pass

        if not pipeline_data.get('documents'):
            print("No content available for further processing.")
            return {}

        structured_data = self.stage_1.process(pipeline_data)
        enriched_data = self.stage_2.process(structured_data)
        synthesized_data = self.stage_3.process(enriched_data)

        final_nodes = synthesized_data.get("primary_nodes", [])

        # Fallback in case Stage 2 (enrich) fails but Stage 1 (structure) worked
        if not final_nodes and structured_data.get('documents'):
            click.echo("  > Finalizing nodes from Stage 1 data...")
            final_nodes = self._create_nodes_from_docs(structured_data.get('documents'))

        # Apply final metadata and prepare for embedding
        if final_nodes:
            final_nodes = self._apply_and_prepare_nodes(final_nodes)
            synthesized_data["primary_nodes"] = final_nodes

        print("--- Cognitive Architect Pipeline Finished ---")
        return synthesized_data

    def _create_nodes_from_docs(self, docs):
        """Helper to convert LlamaIndex Documents to TextNodes."""
        nodes = []
        for doc in docs:
            node = TextNode(
                text=doc.text,
                metadata=doc.metadata
            )
            nodes.append(node)
        return nodes

    def _apply_and_prepare_nodes(self, nodes: list):
        """
        Applies document-level metadata (like summary, filename) to all nodes
        and constructs the final text content to be embedded.
        """
        prepared_nodes = []
        for node in nodes:
            # 1. Store the original, clean chunk text in metadata
            original_text = node.get_content()
            node.metadata['original_text'] = original_text

            # 2. Add file-level metadata (original_filename, file_path)
            node.metadata.update(self.current_file_metadata)

            # 3. Add hash
            node.metadata['hash'] = hashlib.md5(original_text.encode()).hexdigest()

            # 4. Build the new "searchable text"
            searchable_parts = [original_text]

            # Dynamically build searchable content based on config
            for key in self.embeddable_keys:
                if node.metadata.get(key):
                    value_str = str(node.metadata.get(key)).strip()
                    if value_str:
                        searchable_parts.append(f"{key.replace('_', ' ').title()}: {value_str}")

            # 5. Set the node's main content to this new searchable string
            node.set_content("\n\n".join(searchable_parts))

            # 6. Set exclusions for embedding and LLM
            node.excluded_embed_metadata_keys = list(node.metadata.keys())
            node.excluded_llm_metadata_keys = list(node.metadata.keys())

            prepared_nodes.append(node)

        return prepared_nodes