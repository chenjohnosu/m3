import click
from core.project_manager import ProjectManager
from core.llm_dialogue import start_dialogue


@click.group()
@click.pass_context
def project(ctx):
    """Commands for managing projects."""
    # 1. Check if we are in a session (interactive mode)
    #    In m3.py, interactive mode passes an M3Session object.
    #    Standalone mode passes an empty dict {}.
    if ctx.obj:
        # We have a session (or at least a truthy object)
        pass
    else:
        # Fallback for non-interactive mode (e.g., 'm3 project list')
        # We inject a temporary ProjectManager so commands can function.
        ctx.obj = ProjectManager()


def get_manager(ctx):
    """
    Helper to retrieve the ProjectManager instance.
    Handles both M3Session (interactive) and direct ProjectManager (standalone).
    """
    if isinstance(ctx.obj, ProjectManager):
        return ctx.obj
    elif hasattr(ctx.obj, 'project_manager'):
        return ctx.obj.project_manager
    else:
        # Fallback if something went wrong with context initialization
        return ProjectManager()


@project.command('create')
@click.argument('project_name')
@click.pass_context
def create(ctx, project_name):
    """Creates a new, empty project."""
    manager = get_manager(ctx)

    path, message = manager.init_project(project_name)
    if not path:
        click.secho(f"Error: {message}", fg="red")
    else:
        click.secho(f"Success: {message}", fg="green")

        # --- SESSION UPDATE ---
        # If we are in interactive mode, tell the session to load this new project immediately
        if hasattr(ctx.obj, 'load_project'):
            ctx.obj.load_project(project_name)


@project.command('list')
@click.pass_context
def list_projects(ctx):
    """Lists all available projects."""
    manager = get_manager(ctx)

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
    manager = get_manager(ctx)

    success, message = manager.set_active_project(project_name)
    if success:
        click.secho(f"Success: {message}", fg="green")

        # --- SESSION UPDATE ---
        # Tell the session to switch context (re-init Vector/Analyze managers)
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

    manager = get_manager(ctx)

    # Check if we are deleting the currently active project
    active_project, _ = manager.get_active_project()
    is_active = (active_project == project_name)

    success, message = manager.remove_project(project_name)
    if success:
        click.secho(f"Success: {message}", fg="green")

        # --- SESSION UPDATE ---
        # If we deleted the active project, unload it from the session
        if is_active and hasattr(ctx.obj, 'load_project'):
            ctx.obj.load_project(None)
    else:
        click.secho(f"Error: {message}", fg="red")


@project.command('dialogue')
@click.argument('project_name', required=False)
@click.pass_context
def dialogue(ctx, project_name):
    """
    Starts an interactive chat session with the project data.
    This mode maintains conversation history and uses the vector store for context.
    """

    # 1. Ensure we have a valid session (must be run with --go)
    #    The dialogue mode relies on the persistent AnalyzeManager in the session.
    if not hasattr(ctx.obj, 'analyze_manager'):
        click.secho("Error: Dialogue mode requires an active interactive session (run with --go).", fg="red")
        return

    # 2. Handle project switching if an argument is provided
    if project_name:
        # If the user requested a different project, load it now
        if ctx.obj.active_project_name != project_name:
            ctx.obj.load_project(project_name)

    # 3. Validation
    if not ctx.obj.active_project_name:
        click.secho("Error: No active project selected. Use '/project active <name>' first.", fg="red")
        return

    # 4. Check for AnalyzeManager
    #    If the project was loaded but AnalyzeManager failed (e.g. no DB), we can't chat.
    if not ctx.obj.analyze_manager:
        click.secho("Error: AnalyzeManager is not initialized.", fg="red")
        click.secho("  > Try running '/corpus ingest' to build the vector store first.", fg="yellow")
        return

    # 5. Start the Chat Loop
    #    We pass the ready-to-go manager to the dialogue loop
    start_dialogue(ctx.obj.analyze_manager)