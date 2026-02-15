import click
import logging
from dataclasses import dataclass, field
from core.ingestion.base_pipeline import BasePipeline
from core.ingestion.stages.cogarc_stage_0_stratify import CogArcStage0Stratify
from core.ingestion.stages.cogarc_stage_1_structure import CogArcStage1Structure
from core.ingestion.stages.cogarc_stage_2_enrich import CogArcStage2Enrich
from core.ingestion.stages.cogarc_stage_3_synthesis import CogArcStage3Synthesis
from core.llm_manager import LLMManager
# --- NEW: Import TextNode to create nodes ---
from llama_index.core.schema import TextNode
import hashlib

logger = logging.getLogger(__name__)


# --------------------------------------------

@dataclass
class StageRegistration:
    """Metadata for a registered external pipeline stage."""
    name: str
    stage_fn: object            # callable: (data: dict) -> dict
    position: str               # "append" | "after:<stage_name>" | "before:<stage_name>"
    enabled: bool = True
    description: str = ""


class CognitiveArchitectPipeline(BasePipeline):

    # --- MODIFIED: Accept llm_manager in the constructor ---
    def __init__(self, config, llm_manager: LLMManager):
        super().__init__(config)
        print("Initializing Cognitive Architect Pipeline...")
        # self.llm_manager = LLMManager(config) # <-- REMOVED: No longer create a new one
        self.llm_manager = llm_manager          # <-- ADDED: Use the persistent one
        # --- END MODIFIED ---

        self.cogarc_settings = config.get('ingestion_config', {}).get('cogarc_settings', {})

        analysis_config = self.config.get('analysis_settings', {})
        self.embeddable_keys = analysis_config.get('metadata_keys_to_embed', ['themes'])
        click.echo(f"  > CogArc Pipeline: Embedding metadata keys = {self.embeddable_keys}", err=True)

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

        # Assign names to built-in stages for position references
        self.stage_0.name = "stratify"
        self.stage_1.name = "structure"
        self.stage_2.name = "enrich"
        self.stage_3.name = "synthesize"

        self.current_file_metadata = {}
        self._registered_stages = []

    # ------------------------------------------------------------------
    # Stage Registration API
    # ------------------------------------------------------------------

    def register_stage(self, name, stage_fn, position="append", description=""):
        """
        Register an external pipeline stage.

        Args:
            name:        Unique name for this stage (used in logging and position refs).
            stage_fn:    Callable that accepts a dict and returns a dict.
                         Signature: (data: dict) -> dict
            position:    Where to insert the stage:
                           "append"             — after all built-in stages (default)
                           "after:stratify"     — immediately after the named stage
                           "before:synthesize"  — immediately before the named stage
            description: Human-readable description (appears in pipeline logs).

        Raises:
            ValueError: If name is already registered or position target is invalid.
        """
        if any(r.name == name for r in self._registered_stages):
            raise ValueError(f"Stage '{name}' is already registered.")

        if position != "append":
            prefix, _, target = position.partition(":")
            if prefix not in ("after", "before"):
                raise ValueError(
                    f"Invalid position '{position}'. "
                    "Use 'append', 'after:<stage>', or 'before:<stage>'."
                )
            valid_targets = [s.name for s in self._get_builtin_stages()]
            valid_targets += [r.name for r in self._registered_stages]
            if target not in valid_targets:
                raise ValueError(
                    f"Position target '{target}' not found. "
                    f"Available stages: {valid_targets}"
                )

        self._registered_stages.append(
            StageRegistration(
                name=name,
                stage_fn=stage_fn,
                position=position,
                description=description,
            )
        )
        logger.info(f"Registered external stage '{name}' at position '{position}'.")

    def unregister_stage(self, name):
        """Remove a previously registered external stage."""
        before = len(self._registered_stages)
        self._registered_stages = [r for r in self._registered_stages if r.name != name]
        if len(self._registered_stages) == before:
            raise ValueError(f"Stage '{name}' not found in registered stages.")

    def list_stages(self):
        """
        Return ordered list of all stages (built-in + registered) as they will execute.
        """
        result = []
        for s in self._resolve_stage_order():
            if isinstance(s, StageRegistration):
                result.append({
                    "name": s.name,
                    "type": "registered",
                    "position": s.position,
                    "description": s.description,
                    "enabled": s.enabled,
                })
            else:
                result.append({
                    "name": getattr(s, "name", "unknown"),
                    "type": "builtin",
                    "position": "builtin",
                    "description": "",
                    "enabled": True,
                })
        return result

    def _get_builtin_stages(self):
        """Return the list of built-in stages in their default order."""
        return [self.stage_0, self.stage_1, self.stage_2, self.stage_3]

    def _resolve_stage_order(self):
        """
        Merge built-in stages with registered external stages respecting position directives.
        Returns ordered list of stage objects and StageRegistration instances.
        """
        result = list(self._get_builtin_stages())

        for reg in self._registered_stages:
            if not reg.enabled:
                continue
            if reg.position == "append":
                result.append(reg)
            else:
                prefix, _, target = reg.position.partition(":")
                target_idx = next(
                    (i for i, s in enumerate(result)
                     if getattr(s, "name", None) == target),
                    None
                )
                if target_idx is None:
                    logger.warning(
                        f"Stage '{reg.name}': position target '{target}' not found "
                        "at execution time — appending instead."
                    )
                    result.append(reg)
                elif prefix == "after":
                    result.insert(target_idx + 1, reg)
                else:  # before
                    result.insert(target_idx, reg)

        return result

    # ------------------------------------------------------------------
    # Pipeline Execution
    # ------------------------------------------------------------------

    def run(self, documents, doc_type):
        print(f"\n--- Starting Cognitive Architect Pipeline for doc_type: '{doc_type}' ---")

        if documents:
            self.current_file_metadata = documents[0].metadata.copy()
            self.current_file_metadata.pop('text', None)

        pipeline_data = {'documents': documents}

        # Resolve stage order (built-in + registered)
        stages = self._resolve_stage_order()

        for stage in stages:
            if isinstance(stage, StageRegistration):
                # External registered stage
                stage_name = stage.name
                logger.info(f"Running registered stage: {stage_name}")
                print(f"  Running registered stage: {stage_name}")
                pipeline_data = stage.stage_fn(pipeline_data)
            else:
                # Built-in stage
                stage_name = getattr(stage, "name", "unknown")

                # Preserve conditional Stage 0 logic (interview-only)
                if stage_name == "stratify":
                    if doc_type == 'interview':
                        pipeline_data = stage.process(pipeline_data)
                    else:
                        print("Skipping Stage 0 (Q&A Stratification) for non-interview document.")
                    # Check if we still have documents after stratification
                    if not pipeline_data.get('documents'):
                        print("No content available for further processing.")
                        return {}
                else:
                    pipeline_data = stage.process(pipeline_data)

        final_nodes = pipeline_data.get("primary_nodes", [])

        if not final_nodes and pipeline_data.get('documents'):
            click.echo("  > Finalizing nodes from Stage 1 data...")
            final_nodes = self._create_nodes_from_docs(pipeline_data.get('documents'))

        if final_nodes:
            final_nodes = self._apply_and_prepare_nodes(final_nodes)
            pipeline_data["primary_nodes"] = final_nodes

        print("--- Cognitive Architect Pipeline Finished ---")
        return pipeline_data

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
        Applies document-level metadata (like summary) to all nodes
        and constructs the final text to be embedded.
        """
        prepared_nodes = []
        for node in nodes:
            original_text = node.get_content()
            node.metadata['original_text'] = original_text

            node.metadata.update(self.current_file_metadata)

            node.metadata['hash'] = hashlib.md5(original_text.encode()).hexdigest()

            searchable_parts = [original_text]

            for key in self.embeddable_keys:
                if node.metadata.get(key):
                    value_str = str(node.metadata.get(key)).strip()
                    if value_str:
                        searchable_parts.append(f"{key.replace('_', ' ').title()}: {value_str}")

            if self.cogarc_settings.get('include_summary_in_embedding', False):
                if "holistic_summary" in node.metadata:
                    searchable_parts.append(f"Summary: {node.metadata['holistic_summary']}")

            node.set_content("\n\n".join(searchable_parts))

            node.excluded_embed_metadata_keys = list(node.metadata.keys())
            node.excluded_llm_metadata_keys = list(node.metadata.keys())

            prepared_nodes.append(node)

        return prepared_nodes
