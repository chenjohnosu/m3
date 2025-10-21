import click
from core.vector_manager import VectorManager
from utils.config import get_config  # <-- Make sure this is imported


@click.group()
def vector():
    """Commands for managing the project's vector store."""
    pass


@vector.command('ingest')
@click.confirmation_option(
    prompt='This command will re-process the entire corpus, which can be time-consuming.\nAre you sure you want to proceed?')
def ingest():
    """
    (Re)Builds the entire vector store from all files in the corpus.
    This will delete all existing vectors and re-process every file.
    """
    try:
        manager = VectorManager(get_config())  # <-- FIXED
        manager.rebuild_vector_store()
    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")


@vector.command('chunks')
@click.argument('identifier')
@click.option('--meta', 'include_metadata', is_flag=True, help='Show simple metadata.')
@click.option('--pretty', is_flag=True, help='Pretty-print all metadata and content.')
@click.option('--summary', 'show_summary', is_flag=True, help='Explicitly show the holistic summary.')
def chunks(identifier, include_metadata, pretty, show_summary):
    """
    Shows all text chunks for a specific file.

    Identifier can be the file's original name or its corpus ID.
    Example: /vector chunks my_file.txt --pretty
    """
    try:
        manager = VectorManager(get_config())  # <-- FIXED
        # Pass the new show_summary flag to the manager
        manager.get_file_chunks(identifier, include_metadata, pretty, show_summary)
    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")


@vector.command('status')
def status():
    """Displays the status of the vector store."""
    try:
        manager = VectorManager(get_config())  # <-- FIXED
        manager.get_vector_store_status()
    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")


@vector.command('rebuild')
@click.confirmation_option(prompt='This will delete all vectors and re-process all files. Are you sure?')
def rebuild():
    """DEPRECATED. Use the 'ingest' command instead."""
    click.secho("  > This command is deprecated. Forwarding to 'ingest'...", dim=True)
    try:
        manager = VectorManager(get_config())  # <-- FIXED
        manager.rebuild_vector_store()
    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")


@vector.command('create')
@click.confirmation_option(prompt='This will delete the existing vector store. Are you sure?')
def create():
    """Creates a new, blank vector store for the project."""
    try:
        manager = VectorManager(get_config())  # <-- FIXED
        manager.create_vector_store(rebuild=True)
    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")


@vector.command('query')
@click.argument('query_text', nargs=-1)
def query(query_text):
    """Queries the vector store (simple RAG)."""
    if not query_text:
        click.echo("  > Please provide a query string.")
        return
    try:
        manager = VectorManager(get_config())  # <-- FIXED
        manager.query_vector_store(" ".join(query_text))
    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")