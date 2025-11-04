import click
from core.vector_manager import VectorManager
from utils.config import get_config


@click.group()
def vector():
    """Commands for managing the project's vector store."""
    pass


def _get_manager(ctx):
    """Helper to get manager from session or create new."""
    if ctx.obj and hasattr(ctx.obj, 'vector_manager'):
        manager = ctx.obj.vector_manager
        if not manager:
            click.secho("Error: No active project. Please use '/project active <name>'.", fg="red")
            return None
        return manager
    else:
        click.secho("  > (Single Command Mode) Initializing VectorManager...", dim=True)
        try:
            return VectorManager(get_config())
        except Exception as e:
            click.secho(f"Error: {e}", fg="red")
            return None

@vector.command('ingest')
@click.confirmation_option(
    prompt='This command will re-process the entire corpus, which can be time-consuming.\nAre you sure you want to proceed?')
@click.pass_context
def ingest(ctx):
    """
    (Re)Builds the entire vector store from all files in the corpus.
    """
    try:
        manager = _get_manager(ctx)
        if manager:
            manager.rebuild_vector_store()
    except Exception as e:
        click.secho(f"櫨 Error: {e}", fg="red")


@vector.command('chunks')
@click.argument('identifier')
@click.option('--meta', 'include_metadata', is_flag=True, help='Show simple metadata.')
@click.option('--pretty', is_flag=True, help='Pretty-print all metadata and content.')
@click.option('--summary', 'show_summary', is_flag=True, help='Explicitly show the holistic summary.')
@click.pass_context
def chunks(ctx, identifier, include_metadata, pretty, show_summary):
    """
    Shows all text chunks for a specific file.
    """
    try:
        manager = _get_manager(ctx)
        if manager:
            manager.get_file_chunks(identifier, include_metadata, pretty, show_summary)
    except Exception as e:
        click.secho(f"櫨 Error: {e}", fg="red")


@vector.command('status')
@click.pass_context
def status(ctx):
    """Displays the status of the vector store."""
    try:
        manager = _get_manager(ctx)
        if manager:
            manager.get_vector_store_status()
    except Exception as e:
        click.secho(f"櫨 Error: {e}", fg="red")


@vector.command('rebuild')
@click.confirmation_option(prompt='This will delete all vectors and re-process all files. Are you sure?')
@click.pass_context
def rebuild(ctx):
    """DEPRECATED. Use the 'ingest' command instead."""
    click.secho("  > This command is deprecated. Forwarding to 'ingest'...", dim=True)
    try:
        manager = _get_manager(ctx)
        if manager:
            manager.rebuild_vector_store()
    except Exception as e:
        click.secho(f"櫨 Error: {e}", fg="red")


@vector.command('create')
@click.confirmation_option(prompt='This will delete the existing vector store. Are you sure?')
@click.pass_context
def create(ctx):
    """Creates a new, blank vector store for the project."""
    try:
        manager = _get_manager(ctx)
        if manager:
            manager.create_vector_store(rebuild=True)
    except Exception as e:
        click.secho(f"櫨 Error: {e}", fg="red")


@vector.command('query')
@click.argument('query_text', nargs=-1)
@click.pass_context
def query(ctx, query_text):
    """Queries the vector store (simple RAG)."""
    if not query_text:
        click.echo("  > Please provide a query string.")
        return
    try:
        manager = _get_manager(ctx)
        if manager:
            manager.query_vector_store(" ".join(query_text))
    except Exception as e:
        click.secho(f"櫨 Error: {e}", fg="red")