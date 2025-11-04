import click
from core.project_manager import ProjectManager

@click.group()
@click.pass_context
def project(ctx):
    """Commands for managing projects."""
    # Ensure the session is available if in interactive mode
    if ctx.obj:
        pass # Session is already loaded
    else:
        # Fallback for non-interactive mode (e.g., m3 project list)
        ctx.obj = ProjectManager()


@project.command('create')
@click.argument('project_name')
@click.pass_context
def create(ctx, project_name):
    """Creates a new, empty project."""

    # Get manager from session or new instance
    manager = ctx.obj if isinstance(ctx.obj, ProjectManager) else ctx.obj.project_manager

    path, message = manager.init_project(project_name)
    if not path:
        click.secho(f"Error: {message}", fg="red")
    else:
        click.secho(f"Success: {message}", fg="green")
        # --- OPTIMIZATION ---
        # If in interactive mode, tell the session to load this new project
        if hasattr(ctx.obj, 'load_project'):
            ctx.obj.load_project(project_name)


@project.command('list')
@click.pass_context
def list_projects(ctx):
    """Lists all available projects."""
    # Get manager from session or new instance
    manager = ctx.obj if isinstance(ctx.obj, ProjectManager) else ctx.obj.project_manager

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
@click.pass_context
def active(ctx, project_name):
    """Sets the active project for the current session."""
    # Get manager from session or new instance
    manager = ctx.obj if isinstance(ctx.obj, ProjectManager) else ctx.obj.project_manager

    success, message = manager.set_active_project(project_name)
    if success:
        click.secho(f"Success: {message}", fg="green")
        # --- OPTIMIZATION ---
        # If in interactive mode, tell the session to load this project
        if hasattr(ctx.obj, 'load_project'):
            ctx.obj.load_project(project_name)
    else:
        click.secho(f"Error: {message}", fg="red")


@project.command('remove')
@click.argument('project_name')
@click.pass_context
def remove(ctx, project_name):
    """Permanently deletes a project and all its data."""
    click.confirm(f"Are you sure you want to permanently delete the project '{project_name}' and all its data?",
                  abort=True)

    # Get manager from session or new instance
    manager = ctx.obj if isinstance(ctx.obj, ProjectManager) else ctx.obj.project_manager

    active_project, _ = manager.get_active_project()

    success, message = manager.remove_project(project_name)
    if success:
        click.secho(f"Success: {message}", fg="green")
        # --- OPTIMIZATION ---
        # If we deleted the active project, tell the session to load 'None'
        if hasattr(ctx.obj, 'load_project') and active_project == project_name:
            ctx.obj.load_project(None)
    else:
        click.secho(f"Error: {message}", fg="red")


@project.command('dialogue')
@click.argument('project_name')
def dialogue(project_name):
    """(Placeholder) Starts an interactive dialogue with project data."""
    click.echo(f"Starting dialogue for project '{project_name}'... (Not yet implemented)")