"""
DEPRECATED: The 'vector' command group has been renamed to 'index'.
This module exists for backwards compatibility only.
  vector ingest  → index build
  vector status  → index status
  vector chunks  → index chunks
"""
import click
from utils.config import get_config


def _get_manager(ctx):
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


@click.group(hidden=True)
def vector():
    """DEPRECATED. Use 'index' instead."""
    click.secho(
        "  Warning: The 'vector' group is deprecated. Use 'index' instead (/i in interactive mode).",
        fg="yellow", err=True
    )


@vector.command('ingest', hidden=True)
@click.pass_context
def ingest(ctx):
    """DEPRECATED. Use 'index build'."""
    click.secho("  Warning: 'vector ingest' is deprecated. Use 'index build'.", fg="yellow", err=True)
    try:
        click.echo("This will re-process the entire corpus.")
        click.confirm("Are you sure?", abort=True, default=False)
        manager = _get_manager(ctx)
        if manager:
            manager.rebuild_vector_store()
    except click.exceptions.Abort:
        click.echo("Operation cancelled.")
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")


@vector.command('status', hidden=True)
@click.pass_context
def status(ctx):
    """DEPRECATED. Use 'index status'."""
    click.secho("  Warning: 'vector status' is deprecated. Use 'index status'.", fg="yellow", err=True)
    try:
        manager = _get_manager(ctx)
        if manager:
            manager.get_vector_store_status()
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")


@vector.command('chunks', hidden=True)
@click.argument('identifier')
@click.option('--meta', 'include_metadata', is_flag=True)
@click.option('--pretty', is_flag=True)
@click.option('--summary', 'show_summary', is_flag=True)
@click.pass_context
def chunks(ctx, identifier, include_metadata, pretty, show_summary):
    """DEPRECATED. Use 'index chunks'."""
    click.secho("  Warning: 'vector chunks' is deprecated. Use 'index chunks'.", fg="yellow", err=True)
    try:
        manager = _get_manager(ctx)
        if manager:
            manager.get_file_chunks(identifier, include_metadata, pretty, show_summary)
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")


@vector.command('rebuild', hidden=True)
@click.pass_context
def rebuild(ctx):
    """DEPRECATED. Use 'index build'."""
    click.secho("  Warning: 'vector rebuild' is deprecated. Use 'index build'.", fg="yellow", err=True)
    try:
        click.echo("This will re-process the entire corpus.")
        click.confirm("Are you sure?", abort=True, default=False)
        manager = _get_manager(ctx)
        if manager:
            manager.rebuild_vector_store()
    except click.exceptions.Abort:
        click.echo("Operation cancelled.")
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")


@vector.command('create', hidden=True)
@click.pass_context
def create(ctx):
    """DEPRECATED. The index is managed automatically by 'index build'."""
    click.secho(
        "  Warning: 'vector create' is deprecated and no longer needed. Use 'index build'.",
        fg="yellow", err=True
    )


@vector.command('query', hidden=True)
@click.argument('query_text', nargs=-1)
@click.pass_context
def query(ctx, query_text):
    """DEPRECATED. Use 'analyze search'."""
    click.secho("  Warning: 'vector query' is deprecated. Use 'analyze search <query>'.", fg="yellow", err=True)
