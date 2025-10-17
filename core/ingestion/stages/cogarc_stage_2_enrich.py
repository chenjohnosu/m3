from core.ingestion.stages.base_stage import BaseStage
from llama_index.core.node_parser import SentenceSplitter


class CogArcStage2Enrich(BaseStage):
    def process(self, data):
        print(f"Executing CogArc Stage 2: Micro-Context Enrichment using LLM: {self.llm.model}")

        # This stage now reliably receives the documents from Stage 1
        documents_to_chunk = data.get('documents', [])
        if not documents_to_chunk:
            print("  > No documents to process for Stage 2.")
            return data

        # Use the reliable SentenceSplitter to create the final text chunks.
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)

        nodes = splitter.get_nodes_from_documents(documents_to_chunk)

        # TODO: Implement LLM-based enrichment (hypothetical questions, etc.) on these nodes.

        # Assign the final chunks to 'primary_nodes' for the vector store.
        data['primary_nodes'] = nodes

        print(f"  > Generated {len(nodes)} text chunks for enrichment.")
        return data