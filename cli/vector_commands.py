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
def chunks(identifier):
    """Retrieves and displays text chunks and their metadata for a file."""
    try:
        manager = VectorManager()
        # This function in the manager now needs to be updated to return the data
        # instead of printing it directly, or we can just call it as is if it prints.
        # For this example, we assume we need to modify the VectorManager or
        # that the get_file_chunks function will be updated to show metadata.
        # Let's update the call here to reflect the change to print metadata.
        manager.get_file_chunks(identifier, True)

    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")