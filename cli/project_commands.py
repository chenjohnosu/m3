import click
from ..core import project_manager, llm_dialogue

@click.group()
def project():
    """Commands for managing projects."""
    pass

@project.command('create')
@click.argument('project_name')
def create(project_name):
    """Creates a new, empty project."""
    project_manager.create_project(project_name)

@project.command('list')
def list_projects():
    """Lists all available projects."""
    project_manager.list_projects()

@project.command('add')
@click.argument('project_name')
@click.option('-f', '--file-path', required=True, help='Path to the data file.')
def add(project_name, file_path):
    """Adds a data file to the specified project."""
    project_manager.add_file_to_project(project_name, file_path)

@project.command('remove')
@click.argument('project_name')
@click.option('-f', '--file-name', required=True, help='Name of the file to remove.')
def remove(project_name, file_name):
    """Removes a data file from the specified project."""
    project_manager.remove_file_from_project(project_name, file_name)

@project.command('status')
@click.argument('project_name')
def status(project_name):
    """Displays detailed information about a project."""
    project_manager.get_project_status(project_name)

@project.command('dialogue')
@click.argument('project_name')
def dialogue(project_name):
    """Enters the interactive dialogue mode for a project."""
    llm_dialogue.start_dialogue_session(project_name)

@project.command('delete')
@click.argument('project_name')
def delete(project_name):
    """Permanently deletes a project and its associated data."""
    project_manager.delete_project(project_name)
