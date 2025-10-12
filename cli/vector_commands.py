import click
from core import vector_manager

@click.group()
def vector():
    """Commands for managing vector stores associated with projects."""
    pass

@vector.command('list')
def list_stores():
    """Lists all available vector stores."""
    vector_manager.list_vector_stores()

@vector.command('status')
@click.argument('project_name')
def status(project_name):
    """Provides metadata about the vector store for a project."""
    vector_manager.get_vector_store_status(project_name)

@vector.command('rebuild')
@click.argument('project_name')
def rebuild(project_name):
    """Forces a complete rebuild of the vector store from source files."""
    vector_manager.rebuild_vector_store(project_name)

@vector.command('delete')
@click.argument('project_name')
def delete(project_name):
    """Deletes the vector store associated with a project."""
    vector_manager.delete_vector_store(project_name)
