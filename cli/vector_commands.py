import click
from core import vector_manager

@click.group()
def vector():
    """Commands for managing a project's vector store."""
    pass

@vector.command("status")
@click.argument('project_name')
def status(project_name):
    """Gets the status of the vector store."""
    vector_manager.get_status(project_name)

@vector.command("rebuild")
@click.argument('project_name')
def rebuild(project_name):
    """Rebuilds the vector store from the corpus."""
    vector_manager.rebuild(project_name)
