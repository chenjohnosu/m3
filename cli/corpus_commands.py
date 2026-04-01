import glob as glob_module
import click
from utils.config import get_config
from pathlib import Path
import textwrap


# ── Helper ─────────────────────────────────────────────────────────────────

def _get_manager(ctx):
    """Return VectorManager from session or create a fresh one."""
    if ctx.obj and hasattr(ctx.obj, 'vector_manager'):
        return ctx.obj.vector_manager
    from core.vector_manager import VectorManager
    click.secho("  > (Single Command Mode) Initializing VectorManager...", dim=True)
    return VectorManager(get_config())


def _resolve_paths(raw_paths):
    """Expand glob patterns and validate all paths.

    Returns (resolved, errors) where resolved is a list of existing path
    strings and errors is a list of (pattern, message) tuples for anything
    that didn't match.
    """
    resolved = []
    errors = []
    for raw in raw_paths:
        has_glob = any(c in raw for c in ('*', '?', '['))
        if has_glob:
            matches = glob_module.glob(raw, recursive=True)
            if not matches:
                errors.append((raw, "no files matched glob pattern"))
            else:
                resolved.extend(matches)
        else:
            p = Path(raw)
            if p.exists():
                resolved.append(str(p))
            else:
                errors.append((raw, "path does not exist"))
    return resolved, errors


# ── Command group ──────────────────────────────────────────────────────────

@click.group()
def corpus():
    """Manages the document corpus for the active project."""
    pass


@corpus.command('add')
@click.argument('paths', nargs=-1)
@click.option('--type', 'doc_type',
              type=click.Choice(get_config().get('ingestion_config', {}).get('known_doc_types', ['document'])),
              default=None,
              help='The type of document being added.')
@click.pass_context
def add(ctx, paths, doc_type):
    """Adds one or more files, directories, or glob patterns to the corpus.

    Supports wildcards: /c add "data/*.docx" --type interview
    Recursive globs:    /c add "data/**/*.txt"
    Mixed inputs:       /c add file.pdf "notes/*.md" some_dir/
    """
    if not paths:
        click.echo("Error: No file paths provided.")
        return

    resolved, errors = _resolve_paths(paths)

    for pattern, msg in errors:
        click.secho(f"  Warning: '{pattern}' — {msg}", fg="yellow")

    if not resolved:
        click.secho("Error: No valid paths to add.", fg="red")
        return

    config = get_config()
    if not doc_type:
        doc_type = config.get('ingestion_config', {}).get('default_doc_type', 'document')
        click.echo(f"No --type specified, using default: '{doc_type}'")

    if len(resolved) != len(paths):
        click.echo(f"  > Resolved {len(resolved)} path(s) from {len(paths)} input(s).")

    try:
        manager = _get_manager(ctx)
        if not manager:
            click.secho("Error: No active project. Please use '/project active <name>'.", fg="red")
            return

        manager.add_to_corpus(resolved, doc_type)
        click.secho(f"\n✅ Successfully added and processed {len(resolved)} path(s).", fg="green")
    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")


@corpus.command('remove')
@click.argument('identifier')
@click.pass_context
def remove(ctx, identifier):
    """Removes a file from the corpus by its original filename or ID."""
    try:
        manager = _get_manager(ctx)
        if not manager:
            click.secho("Error: No active project. Please use '/project active <name>'.", fg="red")
            return

        success, message = manager.remove_from_corpus(identifier)
        if success:
            click.secho(f"Success: {message}", fg="green")
        else:
            click.secho(f"Error: {message}", fg="red")
    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")


@corpus.command('list')
@click.pass_context
def list_files(ctx):
    """Lists all files in the active project's corpus."""
    try:
        manager = _get_manager(ctx)
        if not manager:
            click.secho("Error: No active project. Please use '/project active <name>'.", fg="red")
            return

        corpus_items = manager.list_corpus()
        if not corpus_items:
            click.echo("The corpus is currently empty.")
            return

        click.echo("\n--- Corpus Contents ---")
        click.echo(f"{'ID':<38} | {'Chunks':<8} | {'Document Type':<20} | {'Original File'}")
        click.echo("-" * 100)

        sorted_items = sorted(
            corpus_items.items(),
            key=lambda item: (item[1].get('doc_type', 'N/A'), Path(item[1].get('original_path', '')).name)
        )

        for path_in_corpus, meta in sorted_items:
            file_id = Path(path_in_corpus).stem
            doc_type = meta.get('doc_type', 'N/A')
            original_filename = Path(meta.get('original_path', 'Unknown')).name
            chunk_count = manager.get_chunk_count(path_in_corpus)

            click.secho(f"{file_id:<38}", fg="cyan", nl=False)
            click.echo(f" | {chunk_count:<8}", nl=False)
            click.echo(f" | {doc_type:<20}", nl=False)
            click.echo(f" | {original_filename}")
        click.echo("-" * 100)
    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")


@corpus.command('ingest')
@click.pass_context
def ingest(ctx):
    """Rebuilds the entire vector store from the project's corpus."""
    try:
        click.echo("This command will re-process the entire corpus, which can be time-consuming.")
        click.confirm("Are you sure you want to proceed?", abort=True, default=False)

        manager = _get_manager(ctx)
        if not manager:
            click.secho("Error: No active project. Please use '/project active <name>'.", fg="red")
            return

        manager.rebuild_vector_store()
    except click.exceptions.Abort:
        click.echo("Operation cancelled by user.")
    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")


@corpus.command('rebuild')
@click.pass_context
def rebuild(ctx):
    """Alias for 'ingest'. Rebuilds the entire vector store."""
    ctx.invoke(ingest)


@corpus.command('summary')
@click.argument('identifier')
@click.pass_context
def summary(ctx, identifier):
    """Displays the holistic summary for a specific file."""
    try:
        manager = _get_manager(ctx)
        if not manager:
            click.secho("Error: No active project. Please use '/project active <name>'.", fg="red")
            return

        success, content = manager.get_holistic_summary(identifier)

        if not success:
            click.secho(f"🔥 {content}", fg="red")
            return

        original_name = content.get('original_name', 'Unknown')
        holistic_summary = content.get('summary', 'No summary found.')

        click.secho(f"\n--- Holistic Summary for: {original_name} ---", bold=True)
        content_indent = "  "
        wrapped_content = textwrap.fill(
            holistic_summary,
            width=100,
            initial_indent=content_indent,
            subsequent_indent=content_indent
        )
        click.echo(wrapped_content)
        click.secho(f"--- End of Summary ---", bold=True)

    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")


@corpus.command('provenance')
@click.argument('identifier')
@click.pass_context
def provenance(ctx, identifier):
    """Shows full provenance record and version history for a document."""
    try:
        manager = _get_manager(ctx)
        if not manager:
            click.secho("Error: No active project. Please use '/project active <name>'.", fg="red")
            return

        success, data = manager.get_provenance(identifier)
        if not success:
            click.secho(f"Error: {data}", fg="red")
            return

        click.secho(f"\n--- Provenance: {Path(data.get('original_path', '')).name} ---", bold=True)
        fields = [
            ('Corpus ID',        Path(data.get('corpus_path', '')).stem),
            ('Original Path',    data.get('original_path', 'N/A')),
            ('Doc Type',         data.get('doc_type', 'N/A')),
            ('Hash (SHA-256)',   data.get('hash', 'N/A')),
            ('Added At',         data.get('added_at', 'N/A')),
            ('Ingested At',      data.get('ingested_at', 'N/A')),
            ('Pipeline',         data.get('pipeline', 'N/A')),
            ('Chunk Count',      str(data.get('chunk_count', 'N/A'))),
            ('Text Cache',       data.get('text_cache_path', 'N/A')),
        ]
        max_label = max(len(label) for label, _ in fields)
        for label, value in fields:
            click.echo(f"  {label:<{max_label}} : ", nl=False)
            click.secho(value, fg="cyan")

        history = data.get('version_history', [])
        if history:
            click.secho(f"\n  Version History ({len(history)} replacement(s)):", bold=True)
            for idx, entry in enumerate(history, start=1):
                click.secho(f"    [{idx}] Replaced at: {entry.get('replaced_at', 'N/A')}", fg="yellow")
                click.echo(f"         Old hash     : {entry.get('old_hash', 'N/A')}")
                click.echo(f"         Old path     : {entry.get('old_original_path', 'N/A')}")
                click.echo(f"         Old cache    : {entry.get('old_text_cache_path', 'N/A')}")
        else:
            click.echo("\n  No version history (original add).")

        click.secho("--- End of Provenance ---", bold=True)

    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")


@corpus.command('reconstitute')
@click.argument('identifier')
@click.option('--output', 'output_path', default=None, type=click.Path(),
              help='Write reconstituted text to this file instead of stdout.')
@click.option('--from-store', is_flag=True, default=False,
              help='Force reassembly from vector store chunks (ignores text cache).')
@click.pass_context
def reconstitute(ctx, identifier, output_path, from_store):
    """Reconstitutes plain text for a document from its text cache or vector store."""
    try:
        manager = _get_manager(ctx)
        if not manager:
            click.secho("Error: No active project. Please use '/project active <name>'.", fg="red")
            return

        success, content = manager.reconstitute_document(identifier, from_store=from_store)
        if not success:
            click.secho(f"Error: {content}", fg="red")
            return

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            click.secho(f"✅ Reconstituted text written to: {output_path}", fg="green")
        else:
            click.echo(content)

    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")


@corpus.command('find-source')
@click.argument('chunk_id')
@click.pass_context
def find_source(ctx, chunk_id):
    """Looks up the source document for a given vector store chunk ID."""
    try:
        manager = _get_manager(ctx)
        if not manager:
            click.secho("Error: No active project. Please use '/project active <name>'.", fg="red")
            return

        success, data = manager.find_source_by_chunk(chunk_id)
        if not success:
            click.secho(f"Error: {data}", fg="red")
            return

        chunk_idx = data.get('chunk_index')
        chunk_total = data.get('chunk_count')
        position = f"{chunk_idx + 1}/{chunk_total}" if chunk_idx is not None and chunk_total else "N/A"

        click.secho(f"\n--- Chunk Source Lookup ---", bold=True)
        fields = [
            ('Source Document', data.get('original_filename', 'Unknown')),
            ('Corpus ID',       Path(data.get('corpus_path', '')).stem if data.get('corpus_path') else 'N/A'),
            ('Chunk Position',  position),
            ('Doc Type',        data.get('doc_type', 'N/A')),
            ('Ingested At',     data.get('ingested_at', 'N/A')),
            ('Original Path',   data.get('original_path', 'N/A')),
            ('Text Cache',      data.get('text_cache_path', 'N/A')),
        ]
        max_label = max(len(label) for label, _ in fields)
        for label, value in fields:
            click.echo(f"  {label:<{max_label}} : ", nl=False)
            click.secho(str(value), fg="cyan")
        click.secho("--- End ---", bold=True)

    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")
