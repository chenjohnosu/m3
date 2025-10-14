import click
from core import project_manager, vector_manager
from utils.config import get_active_project

@click.group()
def corpus():
    """Commands for managing the active project's corpus."""
    pass

@corpus.command("add")
@click.argument('paths', nargs=-1, required=True)
def add(paths):
    """Adds files or directories to the active project's corpus."""
    active_project = get_active_project()
    if not active_project:
        click.echo("Error: No active project set. Use 'm3 project active <PROJECT_NAME>' first.", err=True)
        return
    project_manager.add_to_corpus(active_project, paths)

@corpus.command("remove")
@click.argument('filename')
def remove(filename):
    """Removes a file from the active project's corpus."""
    active_project = get_active_project()
    if not active_project:
        click.echo("Error: No active project set. Use 'm3 project active <PROJECT_NAME>' first.", err=True)
        return
    project_manager.remove_from_corpus(active_project, filename)

@corpus.command("list")
def list_cmd():
    """Lists all files in the active project's corpus."""
    active_project = get_active_project()
    if not active_project:
        click.echo("Error: No active project set. Use 'm3 project active <PROJECT_NAME>' first.", err=True)
        return
    project_manager.list_corpus(active_project)

@corpus.command("ingest")
def ingest():
    """Processes corpus files and adds them to the vector store."""
    active_project = get_active_project()
    if not active_project:
        click.echo("Error: No active project set. Use 'm3 project active <PROJECT_NAME>' first.", err=True)
        return
    vector_manager.ingest_corpus(active_project)