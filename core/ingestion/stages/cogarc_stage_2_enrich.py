from core.ingestion.stages.base_stage import BaseStage
from llama_index.core.node_parser import SentenceSplitter


class CogArcStage2Enrich(BaseStage):
    def process(self, data):
        print(f"Executing CogArc Stage 2: Micro-Context Enrichment using LLM: {self.llm.model}")

        parent_docs = data.get('parent_docs', [])
        if not parent_docs:
            print("  > No parent documents to process for Stage 2.")
            return data

        # --- CORRECTED CHUNKING LOGIC ---
        # Use the reliable SentenceSplitter to create the final text chunks.
        # This provides more direct control over the chunk size.
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)

        nodes = splitter.get_nodes_from_documents(parent_docs)

        # TODO: Implement LLM-based enrichment (hypothetical questions, etc.) on these nodes.

        # Assign the generated nodes to be stored in the vector store.
        data['primary_nodes'] = nodes

        print(f"  > Generated {len(nodes)} text chunks for enrichment.")
        return data