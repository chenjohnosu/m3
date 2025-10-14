import click
from core import vector_manager
from utils.config import get_active_project

@click.group()
def vector():
    """Commands for managing the active project's vector store."""
    pass

@vector.command("status")
def status():
    """Gets the status of the active project's vector store."""
    active_project = get_active_project()
    if not active_project:
        click.echo("Error: No active project set. Use 'm3 project active <PROJECT_NAME>' first.", err=True)
        return
    vector_manager.get_status(active_project)

@vector.command("rebuild")
def rebuild():
    """Rebuilds the vector store for the active project from its corpus."""
    active_project = get_active_project()
    if not active_project:
        click.echo("Error: No active project set. Use 'm3 project active <PROJECT_NAME>' first.", err=True)
        return
    vector_manager.rebuild(active_project)