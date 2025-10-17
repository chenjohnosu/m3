import os
import unicodedata
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document  # Import the Document class
import click


def read_files(paths):
    """
    Reads a list of files and directories, cleans their text content by removing
    Unicode control characters, and returns a list of Document objects.
    """
    documents = []

    input_files = [p for p in paths if os.path.isfile(p)]
    input_dirs = [p for p in paths if os.path.isdir(p)]

    if not input_files and not input_dirs:
        click.secho("Warning: No valid files or directories found in the provided paths.", fg="yellow")
        return documents

    reader = SimpleDirectoryReader(
        input_dir=input_dirs[0] if input_dirs else None,
        input_files=input_files if input_files else None,
        recursive=True
    )

    try:
        loaded_documents = reader.load_data()

        # --- CORRECTED CLEANING STEP ---
        cleaned_documents = []
        for doc in loaded_documents:
            # Remove all Unicode control characters ('C' category)
            cleaned_text = "".join(ch for ch in doc.text if unicodedata.category(ch)[0] != "C")

            # Escape double quotes to prevent JSON parsing errors in downstream LLM calls
            cleaned_text = cleaned_text.replace('"', '\\"')

            # Create a new Document with the cleaned text and original metadata
            new_doc = Document(text=cleaned_text, metadata=doc.metadata)
            cleaned_documents.append(new_doc)

        click.echo(f"Successfully loaded and cleaned {len(cleaned_documents)} document(s).")
        return cleaned_documents

    except Exception as e:
        click.secho(f"🔥 An error occurred while reading files: {e}", fg="red")
        return []

