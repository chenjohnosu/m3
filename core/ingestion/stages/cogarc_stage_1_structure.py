from core.ingestion.stages.base_stage import BaseStage


class CogArcStage1Structure(BaseStage):
    def process(self, data):
        print(f"Executing CogArc Stage 1: Structural Scaffolding using LLM: {self.llm.model}")
        docs_to_process = data.get('documents', [])
        if not docs_to_process:
            print("  > No documents to process for Stage 1.")
            return data

        # This stage now acts as a pass-through for the documents.
        # The actual chunking is now handled reliably in Stage 2.
        data['parent_docs'] = docs_to_process

        print(f"  > Passing {len(docs_to_process)} documents to the next stage.")
        return data