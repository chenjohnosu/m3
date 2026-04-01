import click
from utils.config import get_config


@click.group()
def index():
    """Commands for managing the project's search index."""
    pass


def _get_manager(ctx):
    """Helper to get manager from session or create new."""
    if ctx.obj and hasattr(ctx.obj, 'vector_manager'):
        manager = ctx.obj.vector_manager
        if not manager:
            click.secho("Error: No active project. Use '/project active <name>'.", fg="red")
            return None
        return manager
    else:
        from core.vector_manager import VectorManager
        click.secho("  > (Single Command Mode) Initializing VectorManager...", dim=True)
        try:
            return VectorManager(get_config())
        except Exception as e:
            click.secho(f"Error: {e}", fg="red")
            return None


@index.command('build')
@click.option('--force', is_flag=True, default=False,
              help='Force a full rebuild even if index appears current.')
@click.pass_context
def build(ctx, force):
    """Builds (or rebuilds) the search index from all corpus files."""
    try:
        click.echo("This will re-process the entire corpus, which can be time-consuming.")
        click.confirm("Are you sure you want to proceed?", abort=True, default=False)
        manager = _get_manager(ctx)
        if manager:
            manager.rebuild_vector_store()
    except click.exceptions.Abort:
        click.echo("Operation cancelled.")
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")


@index.command('status')
@click.pass_context
def status(ctx):
    """Displays the current status of the search index."""
    try:
        manager = _get_manager(ctx)
        if manager:
            manager.get_vector_store_status()
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")


@index.command('chunks')
@click.argument('identifier')
@click.option('--meta', 'include_metadata', is_flag=True, help='Show abbreviated metadata.')
@click.option('--pretty', is_flag=True, help='Pretty-print full metadata and content.')
@click.option('--summary', 'show_summary', is_flag=True, help='Also display the holistic summary.')
@click.pass_context
def chunks(ctx, identifier, include_metadata, pretty, show_summary):
    """Shows all indexed chunks for a specific document."""
    try:
        manager = _get_manager(ctx)
        if manager:
            manager.get_file_chunks(identifier, include_metadata, pretty, show_summary)
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")
