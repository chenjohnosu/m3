import click
from utils.config import get_config


@click.group()
def analyze():
    """Commands for analyzing project data."""
    pass


def _get_manager(ctx):
    """Helper to get manager from session or create new."""
    if ctx.obj and hasattr(ctx.obj, 'analyze_manager'):
        manager = ctx.obj.analyze_manager
        if not manager:
            click.secho("Error: No active project. Use '/project active <name>'.", fg="red")
            return None
        return manager
    else:
        from core.analyze_manager import AnalyzeManager
        click.secho("  > (Single Command Mode) Initializing AnalyzeManager...", dim=True)
        try:
            return AnalyzeManager(get_config())
        except Exception as e:
            click.secho(f"Error: {e}", fg="red")
            return None


# ---------------------------------------------------------------------------
# Search — unified command (replaces topk + search + exact)
# ---------------------------------------------------------------------------

@analyze.command("search")
@click.argument('query_text')
@click.option('--k', default=10, type=int,
              help='Number of results to return (semantic mode). Default: 10.')
@click.option('--threshold', default=None, type=float,
              help='Score threshold mode: return all chunks scoring above this value (0.0–1.0).')
@click.option('--exact', is_flag=True,
              help='Exact string match mode (case-sensitive literal search).')
@click.option('--summary', 'show_summary', is_flag=True,
              help='Also show holistic summaries for result sources.')
@click.pass_context
def search(ctx, query_text, k, threshold, exact, show_summary):
    """
    Search the indexed corpus.

    Default: semantic top-k (returns the K most relevant chunks).
    Use --threshold to return all chunks above a similarity score.
    Use --exact for case-sensitive literal string matching.
    """
    try:
        manager = _get_manager(ctx)
        if not manager:
            return
        if exact:
            manager.perform_exact_search(query_text, show_summary)
        elif threshold is not None:
            manager.perform_threshold_search(query_text, threshold, show_summary)
        else:
            manager.perform_topk_search(query_text, k, show_summary)
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")


# ---------------------------------------------------------------------------
# Plugin subcommands — direct invocation (replaces 'analyze run <plugin>')
# ---------------------------------------------------------------------------

@analyze.command("interpret")
@click.option('--k', default=5, type=int, help='Max chunks per document to consider.')
@click.option('--threshold', default=0.7, type=float, help='Similarity threshold for retrieval.')
@click.pass_context
def interpret(ctx, k, threshold):
    """Synthesizes all holistic summaries into a single meta-summary."""
    try:
        manager = _get_manager(ctx)
        if manager:
            manager.run_plugin('interpret', k=k, threshold=threshold)
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")


@analyze.command("clustering")
@click.option('--k', default=5, type=int, help='Number of clusters.')
@click.option('--save', is_flag=True, default=False,
              help='Persist axial themes back to chunk metadata.')
@click.pass_context
def clustering(ctx, k, save):
    """Hierarchical clustering with LLM axial coding."""
    try:
        manager = _get_manager(ctx)
        if manager:
            manager.run_plugin('clustering', k=k, save=save)
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")


@analyze.command("anomaly")
@click.option('--k', default=5, type=int, help='Number of outliers to surface.')
@click.pass_context
def anomaly(ctx, k):
    """Detects anomalous/outlier documents using IsolationForest."""
    try:
        manager = _get_manager(ctx)
        if manager:
            manager.run_plugin('anomaly', k=k)
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")


@analyze.command("visualize")
@click.pass_context
def visualize(ctx):
    """t-SNE 2D scatter plot of document embeddings coloured by theme."""
    try:
        manager = _get_manager(ctx)
        if manager:
            manager.run_plugin('visualize')
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")


@analyze.command("entity")
@click.argument('query_text')
@click.option('--k', default=5, type=int, help='Number of top chunks to retrieve.')
@click.option('--threshold', default=0.7, type=float, help='Similarity threshold for retrieval.')
@click.option('--types', 'options', default=None,
              help='Comma-separated entity types (e.g., people,places,organisations).')
@click.pass_context
def entity(ctx, query_text, k, threshold, options):
    """Extract named entities from chunks relevant to the query."""
    try:
        manager = _get_manager(ctx)
        if manager:
            manager.run_plugin('entity', query_text=query_text, k=k,
                               threshold=threshold, options=options)
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")


@analyze.command("categorize")
@click.argument('query_text')
@click.option('--k', default=5, type=int, help='Number of top chunks to retrieve.')
@click.option('--threshold', default=0.7, type=float, help='Similarity threshold for retrieval.')
@click.option('--categories', 'options', required=True,
              help='Comma-separated category labels to assign chunks to.')
@click.pass_context
def categorize(ctx, query_text, k, threshold, options):
    """Assign relevant chunks to user-defined categories."""
    try:
        manager = _get_manager(ctx)
        if manager:
            manager.run_plugin('categorize', query_text=query_text, k=k,
                               threshold=threshold, options=options)
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")


@analyze.command("sentiment")
@click.argument('query_text')
@click.option('--k', default=5, type=int, help='Number of top chunks to retrieve.')
@click.option('--threshold', default=0.7, type=float, help='Similarity threshold for retrieval.')
@click.pass_context
def sentiment(ctx, query_text, k, threshold):
    """Analyse sentiment of chunks relevant to the query."""
    try:
        manager = _get_manager(ctx)
        if manager:
            manager.run_plugin('sentiment', query_text=query_text, k=k, threshold=threshold)
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")


@analyze.command("summarize")
@click.argument('query_text')
@click.option('--k', default=5, type=int, help='Number of top chunks to retrieve.')
@click.option('--threshold', default=0.7, type=float, help='Similarity threshold for retrieval.')
@click.pass_context
def summarize(ctx, query_text, k, threshold):
    """Generate a narrative synthesis of chunks relevant to the query."""
    try:
        manager = _get_manager(ctx)
        if manager:
            manager.run_plugin('summarize', query_text=query_text, k=k, threshold=threshold)
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")


# ---------------------------------------------------------------------------
# Plugin listing
# ---------------------------------------------------------------------------

@analyze.command("tools")
@click.pass_context
def tools(ctx):
    """Lists all available analysis plugins."""
    try:
        if ctx.obj and hasattr(ctx.obj, 'plugin_manager'):
            manager = ctx.obj.plugin_manager
        else:
            from core.plugin_manager import PluginManager
            click.secho("  > (Single Command Mode) Initializing PluginManager...", dim=True)
            manager = PluginManager()

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
        click.secho(f"Error: {e}", fg="red")


# ---------------------------------------------------------------------------
# Deprecated aliases (hidden — not shown in --help)
# ---------------------------------------------------------------------------

@analyze.command("topk", hidden=True)
@click.argument('query_text')
@click.option('--k', default=3, type=int)
@click.option('--summary', 'show_summary', is_flag=True)
@click.pass_context
def topk(ctx, query_text, k, show_summary):
    """DEPRECATED. Use 'analyze search --k N'."""
    click.secho(
        "  Warning: 'analyze topk' is deprecated. Use 'analyze search --k N' instead.",
        fg="yellow", err=True
    )
    try:
        manager = _get_manager(ctx)
        if manager:
            manager.perform_topk_search(query_text, k, show_summary)
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")


@analyze.command("exact", hidden=True)
@click.argument('query_text')
@click.option('--summary', 'include_summary', is_flag=True)
@click.pass_context
def exact(ctx, query_text, include_summary):
    """DEPRECATED. Use 'analyze search --exact'."""
    click.secho(
        "  Warning: 'analyze exact' is deprecated. Use 'analyze search --exact' instead.",
        fg="yellow", err=True
    )
    try:
        manager = _get_manager(ctx)
        if manager:
            manager.perform_exact_search(query_text, include_summary)
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")


@analyze.command("run", hidden=True)
@click.argument('plugin_name')
@click.argument('query_text', required=False)
@click.option('--k', default=5, type=int)
@click.option('--threshold', default=0.7, type=float)
@click.option('--options')
@click.option('--save', is_flag=True, default=False)
@click.pass_context
def run(ctx, plugin_name, query_text, k, threshold, options, save):
    """DEPRECATED. Use 'analyze <plugin-name>' directly."""
    click.secho(
        f"  Warning: 'analyze run' is deprecated. Use 'analyze {plugin_name}' directly.",
        fg="yellow", err=True
    )
    try:
        manager = _get_manager(ctx)
        if manager:
            manager.run_plugin(plugin_name, query_text=query_text, k=k,
                               threshold=threshold, options=options, save=save)
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")
