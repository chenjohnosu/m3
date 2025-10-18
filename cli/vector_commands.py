import click
from core.vector_manager import VectorManager

@click.group()
def vector():
    """Commands for interacting with the vector store of the active project."""
    pass

@vector.command('status')
def status():
    """Displays a detailed status of the vector store."""
    try:
        manager = VectorManager()
        manager.get_vector_store_status()
    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")

@vector.command('create')
def create():
    """Creates a new, blank vector store for the active project."""
    try:
        click.confirm("This will permanently delete the existing vector store and its data. Are you sure?", abort=True)
        manager = VectorManager()
        manager.create_vector_store()
    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")


@vector.command('chunks')
@click.argument('identifier')
@click.option('--pretty', is_flag=True, help='Pretty-print chunk metadata as a table.')
def chunks(identifier, pretty):
    """Retrieves and displays text chunks and their metadata for a file."""
    try:
        manager = VectorManager()
        # Pass the new 'pretty' flag to the manager.
        # We still pass include_metadata=True for the default (non-pretty) view.
        manager.get_file_chunks(identifier, include_metadata=True, pretty=pretty)

    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")