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


@vector.command("create")
def create():
    """
    Creates a new, blank vector store for the active project.

    Warning: If a vector store already exists, it will be permanently deleted.
    """
    active_project = get_active_project()
    if not active_project:
        click.echo("Error: No active project set. Use 'm3 project active <PROJECT_NAME>' first.", err=True)
        return

    click.confirm(
        f"Are you sure you want to create a new vector store for '{active_project}'?\n"
        "This will delete the existing store and all its data.",
        abort=True,
        default=False
    )

    click.echo(f"Creating new vector store for '{active_project}'...")
    vector_manager.create_store(active_project)