import json
import click
from core.ingestion.stages.base_stage import BaseStage
from llama_index.core.schema import Document
from llama_index.core.llms import ChatMessage

SYSTEM_PROMPT = """
You are an expert qualitative data analysis assistant. Your task is to read an interview transcript and stratify it into a series of questions and their corresponding answers.
Analyze the provided transcript and perform the following actions:
1.  Identify each question asked by the "Interviewer".
2.  Identify the block of text that constitutes the "Interviewee's" answer to that question.
3.  Structure your output as a single, valid JSON array of objects.
4.  Each object in the array must represent a single answer and contain two keys:
    - "question": A string containing the full text of the question.
    - "answer": A string containing the full, corresponding block of text for the answer.
Important Rules:
-   Ignore any introductory text or metadata at the beginning of the transcript.
-   Combine multi-part answers into a single "answer" block for the most recent question.
-   Ensure the final output is only the JSON array, with no explanations or conversational text.
"""


class CogArcStage0Stratify(BaseStage):
    def process(self, data):
        print(f"Executing CogArc Stage 0: Q&A Stratification using LLM: {self.llm.model}")
        documents = data.get('documents', [])
        stratified_answers = []
        canonical_questions = set()

        for doc in documents:
            try:
                messages = [
                    ChatMessage(role="system", content=SYSTEM_PROMPT),
                    ChatMessage(role="user", content=doc.text)
                ]

                click.echo(
                    f"  > Sending '{doc.metadata.get('file_name')}' to LLM for analysis. This may take a moment...")

                response = self.llm.chat(messages)

                qa_pairs = json.loads(response.message.content)

                for pair in qa_pairs:
                    question = pair.get("question")
                    answer = pair.get("answer")

                    if question and answer:
                        # --- THE FIX ---
                        # Preserve the file_path from the original document in the new answer document's metadata.
                        answer_doc = Document(
                            text=answer,
                            metadata={
                                "file_path": doc.metadata.get("file_path"),
                                "original_filename": doc.metadata.get("file_name", "Unknown"),
                                "question": question
                            }
                        )
                        stratified_answers.append(answer_doc)
                        canonical_questions.add(question)

                click.secho(f"  > Successfully stratified document into {len(qa_pairs)} Q&A pairs.", fg="green")

            except (json.JSONDecodeError, AttributeError) as e:
                click.secho(f"  > Failed to parse LLM response for document. Error: {e}", fg="red")
            except Exception as e:
                click.secho(f"  > An unexpected error occurred during stratification: {e}", fg="red")

        data['documents'] = stratified_answers
        data['questions'] = list(canonical_questions)
        return data