import click
from core import project_manager
from utils.config import get_active_project


@click.group()
def corpus():
    """Commands for managing the active project's corpus."""
    pass


@corpus.command("add")
@click.argument('paths', nargs=-1, required=True)
def add(paths):
    """
    Adds files, directories, or wildcard patterns to the active project's corpus.

    Examples:
      m3 corpus add my_file.txt
      m3 corpus add "data/*.txt"
      m3 corpus add my_folder/
    """
    active_project = get_active_project()
    if not active_project:
        click.echo("Error: No active project set. Use 'm3 project active <PROJECT_NAME>' first.", err=True)
        return
    project_manager.add_to_corpus(active_project, paths)


@corpus.command("remove")
@click.option('-f', '--file', 'filename', required=True, help='Filename to remove from the corpus.')
def remove(filename):
    """Removes a file from the active project's corpus."""
    active_project = get_active_project()
    if not active_project:
        click.echo("Error: No active project set. Use 'm3 project active <PROJECT_NAME>' first.", err=True)
        return
    # Note: This will require implementing a 'remove_from_corpus' function in project_manager.py
    click.echo(f"Placeholder for removing '{filename}' from '{active_project}'")


@corpus.command("list")
def list_cmd():
    """Lists all files in the active project's corpus."""
    active_project = get_active_project()
    if not active_project:
        click.echo("Error: No active project set. Use 'm3 project active <PROJECT_NAME>' first.", err=True)
        return
    # Note: This will require implementing a 'list_corpus' function in project_manager.py
    click.echo(f"Placeholder for listing corpus of '{active_project}'")