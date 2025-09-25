import click
from core import project_manager

@click.group()
def corpus():
    """Commands for managing a project's corpus."""
    pass

@corpus.command("add")
@click.argument('project_name')
@click.option('-f', '--file', 'file_path', required=True, help='Path to the file to add.')
def add(project_name, file_path):
    """Adds a file to the project's corpus."""
    project_manager.add_to_corpus(project_name, file_path)

@corpus.command("remove")
@click.argument('project_name')
@click.option('-f', '--file', 'filename', required=True, help='Filename to remove from the corpus.')
def remove(project_name, filename):
    """Removes a file from the project's corpus."""
    project_manager.remove_from_corpus(project_name, filename)

@corpus.command("list")
@click.argument('project_name')
def list_cmd(project_name):
    """Lists all files in the project's corpus."""
    project_manager.list_corpus(project_name)
