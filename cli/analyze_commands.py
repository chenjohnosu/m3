import click
from core.analyze_manager import AnalyzeManager
from utils.config import get_config
from core.plugin_manager import PluginManager


@click.group()
def analyze():
    """Commands for analyzing project data."""
    pass

def _get_manager(ctx):
    """Helper to get manager from session or create new."""
    if ctx.obj and hasattr(ctx.obj, 'analyze_manager'):
        manager = ctx.obj.analyze_manager
        if not manager:
            click.secho("Error: No active project. Please use '/project active <name>'.", fg="red")
            return None
        return manager
    else:
        click.secho("  > (Single Command Mode) Initializing AnalyzeManager...", dim=True)
        try:
            return AnalyzeManager(get_config())
        except Exception as e:
            click.secho(f"Error: {e}", fg="red")
            return None


@analyze.command("topk")
@click.argument('query_text')
@click.option('--k', default=3, type=int, help='The number of top chunks to retrieve. Default is 3.')
@click.option('--summary', 'show_summary', is_flag=True, help='Also show holistic summaries in results.')
@click.pass_context
def topk(ctx, query_text, k, show_summary):
    """
    Finds the Top-K most semantically aligned chunks.
    """
    try:
        manager = _get_manager(ctx)
        if manager:
            manager.perform_topk_search(query_text, k, show_summary)
    except Exception as e:
        click.secho(f"櫨 Error: {e}", fg="red")


@analyze.command("search")
@click.argument('query_text')
@click.option('--threshold', default=0.7, type=float,
              help='The similarity threshold (0.0 to 1.0). Finds all chunks with a score *greater than* this value.')
@click.option('--summary', 'show_summary', is_flag=True, help='Also show holistic summaries in results.')
@click.pass_context
def search(ctx, query_text, threshold, show_summary):
    """
    (Phase 1) Finds ALL chunks that meet a similarity threshold.
    """
    try:
        manager = _get_manager(ctx)
        if manager:
            manager.perform_threshold_search(query_text, threshold, show_summary)
    except Exception as e:
        click.secho(f"櫨 Error: {e}", fg="red")


@analyze.command("exact")
@click.argument('query_text')
@click.option('--summary', 'include_summary', is_flag=True, help='Also search in and display holistic summaries.')
@click.pass_context
def exact(ctx, query_text, include_summary):
    """
    Finds all chunks with an *exact string match* (case-sensitive).
    """
    try:
        manager = _get_manager(ctx)
        if manager:
            manager.perform_exact_search(query_text, include_summary)
    except Exception as e:
        click.secho(f"櫨 Error: {e}", fg="red")


# --- PLUGIN COMMANDS ---

@analyze.command("tools")
@click.pass_context
def tools(ctx):
    """Lists all available Phase 2 analysis plugins."""
    try:
        # --- MODIFIED ---
        # Get plugin manager from session or create new
        if ctx.obj and hasattr(ctx.obj, 'plugin_manager'):
            manager = ctx.obj.plugin_manager
        else:
            click.secho("  > (Single Command Mode) Initializing PluginManager...", dim=True)
            manager = PluginManager()
        # --- END MODIFIED ---

        click.secho("--- Available Analysis Tools ---", bold=True)
        plugins = manager.get_plugins()

        if not plugins:
            click.secho("No plugins found in the 'plugins' directory.", fg="yellow")
            return

        max_key_len = max(len(key) for key in plugins.keys()) if plugins else 0

        for key, plugin in sorted(plugins.items()):
            click.secho(f"  {key:<{max_key_len}}", fg="cyan", nl=False)
            click.echo(f" : {plugin.description}")

    except Exception as e:
        click.secho(f"櫨 Error: {e}", fg="red")


@analyze.command("run")
@click.argument('plugin_name')
@click.argument('query_text', required=False)
@click.option('--k', default=5, type=int,
              help='Number of items (e.g., clusters, outliers, or top chunks for LLM).')
@click.option('--threshold', default=0.7, type=float,
              help='Similarity threshold for LLM plugins (0.0 to 1.0).')
@click.option('--options', help='Comma-separated options for plugins (e.g., categories).')
@click.option('--save', is_flag=True, default=False, help="Persist analysis results back to metadata (if supported).")
@click.pass_context
def run(ctx, plugin_name, query_text, k, threshold, options, save):
    """
    Runs a specific analysis plugin.
    """
    try:
        # --- MODIFIED ---
        # Get the AnalyzeManager, which holds the pre-loaded PluginManager
        manager = _get_manager(ctx)
        if not manager:
            return # _get_manager already printed an error
        # --- END MODIFIED ---

        # The analyze_manager now handles finding and running the plugin
        manager.run_plugin(
            plugin_name,
            query_text=query_text,
            k=k,
            threshold=threshold,
            options=options,
            save=save
        )
    except Exception as e:
        click.secho(f"櫨 Error: {e}", fg="red")