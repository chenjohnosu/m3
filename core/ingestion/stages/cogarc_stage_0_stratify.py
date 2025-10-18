import json
import re
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
        processed_documents = []  # This list will hold both stratified and fallback documents
        canonical_questions = set()

        for doc in documents:
            try:
                messages = [
                    ChatMessage(role="system", content=SYSTEM_PROMPT),
                    ChatMessage(role="user", content=doc.text)
                ]

                # Get the original_filename from metadata (which was injected by VectorManager)
                doc_display_name = doc.metadata.get('original_filename', doc.metadata.get('file_name', 'Unknown'))

                click.echo(
                    f"  > Sending '{doc_display_name}' to LLM for analysis. This may take a moment...")

                response = self.llm.chat(messages)
                response_text = response.message.content

                # Use regex to robustly find the JSON array
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)

                if not json_match:
                    # If no JSON is found, trigger the fallback
                    raise ValueError("No JSON array found in the LLM response.")

                json_string = json_match.group(0)
                qa_pairs = json.loads(json_string)

                stratified_count = 0
                for pair in qa_pairs:
                    question = pair.get("question")
                    answer = pair.get("answer")

                    if question and answer:
                        # --- THIS IS THE FIX ---
                        # 1. Create a deep copy of the parent document's metadata.
                        #    This preserves 'original_filename' and any other metadata.
                        new_metadata = doc.metadata.copy()

                        # 2. Update the metadata with new/changed values for this chunk.
                        new_metadata["question"] = question

                        # 3. Ensure 'original_filename' is set to the correct display name
                        #    (in case the parent doc's key was just 'file_name')
                        new_metadata["original_filename"] = doc_display_name

                        answer_doc = Document(
                            text=answer,
                            metadata=new_metadata  # Use the new, copied metadata
                        )
                        processed_documents.append(answer_doc)
                        canonical_questions.add(question)
                        stratified_count += 1

                if stratified_count > 0:
                    click.secho(f"  > Successfully stratified document into {stratified_count} Q&A pairs.", fg="green")
                else:
                    # If JSON was valid but had no content, trigger fallback
                    raise ValueError("JSON was valid but contained no processable Q&A pairs.")


            except (json.JSONDecodeError, ValueError, Exception) as e:
                # --- FALLBACK LOGIC ---
                # If any error occurs during stratification, treat it as a regular document.
                click.secho(
                    f"  > Warning: Failed to stratify '{doc.metadata.get('original_filename', 'Unknown')}'. Reason: {e}",
                    fg="yellow")
                click.secho(f"  > Fallback: Treating as a standard document and passing to the next stage.",
                            fg="yellow")
                processed_documents.append(doc)  # Add the original document to the list

        data['documents'] = processed_documents
        data['questions'] = list(canonical_questions)
        return data