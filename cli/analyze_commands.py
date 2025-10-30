import click
from core.analyze_manager import AnalyzeManager
from utils.config import get_config  # <-- IMPORT ADDED


@click.group()
def analyze():
    """Commands for analyzing project data."""
    pass


@analyze.command("topk")
@click.argument('query_text')
@click.option('--k', default=3, type=int, help='The number of top chunks to retrieve. Default is 3.')
@click.option('--summary', 'show_summary', is_flag=True, help='Also show holistic summaries in results.')
def topk(query_text, k, show_summary):
    """
    Finds the Top-K most semantically aligned chunks.

    Searches (by default): Chunk Content + Themes
    (Summary inclusion for search is controlled by 'config.yaml')

    Example: /analyze topk "connection" --k 5
    """
    try:
        manager = AnalyzeManager(get_config())  # <-- MODIFIED
        manager.perform_topk_search(query_text, k, show_summary)
    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")


@analyze.command("search")
@click.argument('query_text')
@click.option('--threshold', default=0.7, type=float,
              help='The similarity threshold (0.0 to 1.0). Finds all chunks with a score *greater than* this value.')
@click.option('--summary', 'show_summary', is_flag=True, help='Also show holistic summaries in results.')
def search(query_text, threshold, show_summary):
    """
    (Phase 1) Finds ALL chunks that meet a similarity threshold.

    Searches (by default): Chunk Content + Themes
    (Summary inclusion for search is controlled by 'config.yaml')

    Example: /analyze search "grounded theory" --threshold 0.8
    """
    try:
        manager = AnalyzeManager(get_config())  # <-- MODIFIED
        manager.perform_threshold_search(query_text, threshold, show_summary)
    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")


@analyze.command("exact")
@click.argument('query_text')
@click.option('--summary', 'include_summary', is_flag=True, help='Also search in and display holistic summaries.')
def exact(query_text, include_summary):
    """
    Finds all chunks with an *exact string match* (case-sensitive).

    Searches (by default): Chunk Content + Themes
    Use --summary to search in summaries as well.

    Example: /analyze exact "professors" --summary
    """
    try:
        manager = AnalyzeManager(get_config())  # <-- MODIFIED
        # This flag controls both search and display
        manager.perform_exact_search(query_text, include_summary)
    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")


# --- PLUGIN COMMANDS ---

@analyze.command("tools")
def tools():
    """Lists all available Phase 2 analysis plugins."""
    try:
        manager = AnalyzeManager(get_config())  # <-- MODIFIED
        manager.list_plugins()
    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")


@analyze.command("run")
@click.argument('plugin_name')
@click.argument('query_text', required=False)
@click.option('--k', default=5, type=int,
              help='Number of items (e.g., clusters, outliers, or top chunks for LLM).')
@click.option('--threshold', default=0.7, type=float,
              help='Similarity threshold for LLM plugins (0.0 to 1.0).')
@click.option('--options', help='Comma-separated options for plugins (e.g., categories).')
# --- NEW: Add the --save flag ---
@click.option('--save', is_flag=True, default=False, help="Persist analysis results back to metadata (if supported).")
# ---------------------------------
def run(plugin_name, query_text, k, threshold, options, save):  # <-- MODIFIED: Added 'save'
    """
    Runs a specific analysis plugin.

    Examples:

    /a run clustering --k 3

    /a run clustering --k 3 --save

    /a run summarize "user connection" --k 5 --threshold 0.75

    /a run entity "safety concerns" --options="People,Locations"
    """
    try:
        manager = AnalyzeManager(get_config())  # <-- MODIFIED
        # Pass all args as keyword arguments
        manager.run_plugin(
            plugin_name,
            query_text=query_text,
            k=k,
            threshold=threshold,
            options=options,
            save=save  # <-- MODIFIED: Pass the 'save' flag
        )
    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")