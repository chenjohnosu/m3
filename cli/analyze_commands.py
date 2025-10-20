import click
from core.analyze_manager import AnalyzeManager


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
        manager = AnalyzeManager()
        manager.perform_topk_search(query_text, k, show_summary)
    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")


@analyze.command("search")
@click.argument('query_text')
@click.option('--threshold', default=1.0, type=float,
              help='The similarity threshold (L2 Distance). Finds all chunks with a score *less than* this value. Default is 1.0.')
@click.option('--summary', 'show_summary', is_flag=True, help='Also show holistic summaries in results.')
def search(query_text, threshold, show_summary):
    """
    (Phase 1) Finds ALL chunks that meet a similarity threshold.

    Searches (by default): Chunk Content + Themes
    (Summary inclusion for search is controlled by 'config.yaml')

    Example: /analyze search "grounded theory" --threshold 0.8
    """
    try:
        manager = AnalyzeManager()
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
        manager = AnalyzeManager()
        # This flag controls both search and display
        manager.perform_exact_search(query_text, include_summary)
    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")