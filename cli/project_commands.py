import click
from core import project_manager, llm_dialogue

@click.group()
def project():
    """Commands for managing projects."""
    pass

@project.command("create")
@click.argument('project_name')
def create(project_name):
    """Creates a new project."""
    project_manager.create_project(project_name)

@project.command("list")
def list_cmd():
    """Lists all available projects."""
    project_manager.list_projects()

@project.command("remove")
@click.argument('project_name')
def remove(project_name):
    """Removes a project and all its data."""
    project_manager.delete_project(project_name)

@project.command("dialogue")
@click.argument('project_name')
def dialogue(project_name):
    """Starts a dialogue session with the project's data."""
    llm_dialogue.start_dialogue(project_name)

@project.command("active")
@click.argument('project_name')
def active(project_name):
    """Sets the active project."""
    project_manager.set_active(project_name)