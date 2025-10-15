import click
from core.project_manager import ProjectManager


@click.group()
def project():
    """Commands for managing projects."""
    pass


@project.command('create')
@click.argument('project_name')
def create(project_name):
    """Creates a new, empty project."""
    manager = ProjectManager()
    path, message = manager.init_project(project_name)
    if not path:
        click.secho(f"Error: {message}", fg="red")
    else:
        click.secho(f"Success: {message}", fg="green")


@project.command('list')
def list_projects():
    """Lists all available projects."""
    manager = ProjectManager()
    projects = manager.list_projects()
    active_project, _ = manager.get_active_project()

    if not projects:
        click.echo("No projects found.")
        return

    click.echo("Available projects:")
    for proj in projects:
        if proj == active_project:
            click.secho(f"  * {proj} (active)", fg="green")
        else:
            click.echo(f"  - {proj}")


@project.command('active')
@click.argument('project_name')
def active(project_name):
    """Sets the active project for the current session."""
    manager = ProjectManager()
    success, message = manager.set_active_project(project_name)
    if success:
        click.secho(f"Success: {message}", fg="green")
    else:
        click.secho(f"Error: {message}", fg="red")


@project.command('remove')
@click.argument('project_name')
def remove(project_name):
    """Permanently deletes a project and all its data."""
    click.confirm(f"Are you sure you want to permanently delete the project '{project_name}' and all its data?",
                  abort=True)
    manager = ProjectManager()
    success, message = manager.remove_project(project_name)
    if success:
        click.secho(f"Success: {message}", fg="green")
    else:
        click.secho(f"Error: {message}", fg="red")


@project.command('dialogue')
@click.argument('project_name')
def dialogue(project_name):
    """(Placeholder) Starts an interactive dialogue with project data."""
    click.echo(f"Starting dialogue for project '{project_name}'... (Not yet implemented)")