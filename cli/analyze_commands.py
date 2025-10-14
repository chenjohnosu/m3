import click
from core import analyze_manager
from utils.config import get_active_project


@click.group()
def analyze():
    """Commands for analyzing project data."""
    pass


@analyze.command("query")
@click.argument('query_term')
@click.option('-p', '--project', 'project_name', default=None,
              help='The project to query. Defaults to the active project.')
def query(query_term, project_name):
    """Finds a term in the vector store and text source."""
    active_project = get_active_project()

    target_project = project_name if project_name is not None else active_project

    if target_project is None:
        click.echo("Error: No project specified and no active project is set.", err=True)
        click.echo("Use 'm3 project active <PROJECT_NAME>' to set an active project.", err=True)
        return

    analyze_manager.query_vector_store(target_project, query_term)