import click
from core.analyze_manager import AnalyzeManager

@click.group()
def analyze():
    """Commands for analyzing project data."""
    pass

@analyze.command("query")
@click.argument('query_term')
def query(query_term):
    """
    Finds a term in the active project's vector store.
    """
    try:
        # The manager now handles finding the active project internally.
        manager = AnalyzeManager()
        manager.query_vector_store(query_term)
    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")