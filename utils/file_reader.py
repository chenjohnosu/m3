import os
from llama_index.core import SimpleDirectoryReader


def read_files(paths):
    """
    Reads a list of files and directories and returns a list of Document objects.

    This function is designed to handle a mixed list of input paths. It will
    recursively load all supported file types from directories and also load
    any individual files specified.

    Args:
        paths (list): A list of strings, where each string is a path to a
                      file or a directory.

    Returns:
        list: A list of LlamaIndex Document objects, ready for processing.
    """
    documents = []

    # Separate the paths into files and directories for the reader
    input_files = [p for p in paths if os.path.isfile(p)]
    input_dirs = [p for p in paths if os.path.isdir(p)]

    if not input_files and not input_dirs:
        print("Warning: No valid files or directories found in the provided paths.")
        return documents

    # Use SimpleDirectoryReader which can handle both files and directories
    # It will recursively search directories for supported file types.
    reader = SimpleDirectoryReader(
        input_dir=input_dirs[0] if input_dirs else None,
        input_files=input_files if input_files else None,
        recursive=True
    )

    # The reader's load_data method returns the Document objects
    try:
        loaded_documents = reader.load_data()
        print(f"Successfully loaded {len(loaded_documents)} document(s).")
        return loaded_documents
    except Exception as e:
        print(f"🔥 An error occurred while reading files: {e}")
        return []