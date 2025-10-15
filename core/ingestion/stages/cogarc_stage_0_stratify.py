from core.ingestion.stages.base_stage import BaseStage

class CogArcStage0Stratify(BaseStage):
    def process(self, data):
        print(f"Executing CogArc Stage 0: Q&A Stratification using LLM: {self.llm.model}")
        documents = data.get('documents', [])
        # --- LLM LOGIC TO BE IMPLEMENTED ---
        # TODO: Implement LLM logic to classify text into questions/answers
        # and canonicalize questions.
        stratified_data = {'answers': documents, 'questions': []}
        return stratified_data