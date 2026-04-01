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
@click.option('--ingest', 'auto_ingest', is_flag=True, default=False,
              help='Automatically build the search index after adding.')
@click.pass_context
def add(ctx, paths, doc_type, auto_ingest):
    """Adds one or more files, directories, or glob patterns to the corpus.

    Supports wildcards: /c add "data/*.docx" --type interview
    Recursive globs:    /c add "data/**/*.txt"
    Mixed inputs:       /c add file.pdf "notes/*.md" some_dir/

    Use --ingest to build the search index immediately after adding.
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
        click.secho(f"\n✅ Successfully added {len(resolved)} path(s).", fg="green")

        if auto_ingest:
            click.echo("\n  > Building search index (--ingest)...")
            manager.rebuild_vector_store()
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


@corpus.command('update')
@click.option('--yes', '-y', 'auto_approve', is_flag=True, default=False,
              help='Automatically approve all updates without prompting.')
@click.option('--type', 'doc_type', default=None,
              help='Document type for newly discovered files. Defaults to project default.')
@click.pass_context
def update(ctx, auto_approve, doc_type):
    """Scan source directories for new and changed files, then re-ingest.

    For each file already in the corpus, checks if the source file has changed
    (by SHA-256 hash). Also scans the same directories for files not yet added.
    Lists all pending changes and prompts before acting, unless --yes is given.
    """
    try:
        manager = _get_manager(ctx)
        if not manager:
            click.secho("Error: No active project. Please use '/project active <name>'.", fg="red")
            return

        click.echo("Scanning source directories for changes...")
        report = manager.scan_for_updates()

        changed = report['changed']
        new     = report['new']
        missing = report['missing']

        if not changed and not new and not missing:
            click.secho("  Corpus is up to date. No changes detected.", fg="green")
            return

        # ── Report missing source files (informational only) ──────────────
        if missing:
            click.secho(f"\n  {len(missing)} source file(s) no longer found on disk (corpus entry kept):",
                        fg="yellow")
            for orig_str, _, _ in missing:
                click.echo(f"    [MISSING]  {Path(orig_str).name}")
                click.echo(f"               {orig_str}")

        # ── Report changed files ───────────────────────────────────────────
        if changed:
            click.secho(f"\n  {len(changed)} file(s) with changed content:", fg="cyan")
            for orig_str, _, meta, new_hash in changed:
                old_hash = meta.get('hash', 'N/A')
                click.secho(f"    [CHANGED]  {Path(orig_str).name}", fg="cyan")
                click.echo(f"               {orig_str}")
                click.echo(f"               hash  {old_hash[:16]}...  →  {new_hash[:16]}...")

        # ── Report new files ───────────────────────────────────────────────
        if new:
            click.secho(f"\n  {len(new)} new file(s) found in source directories:", fg="cyan")
            for orig_str in new:
                click.secho(f"    [NEW]      {Path(orig_str).name}", fg="cyan")
                click.echo(f"               {orig_str}")

        if not changed and not new:
            # Only missing — nothing actionable beyond informing the user
            return

        # ── Confirm ────────────────────────────────────────────────────────
        total = len(changed) + len(new)
        click.echo(f"\n{total} file(s) will be re-ingested.")

        if not auto_approve:
            try:
                click.confirm("Proceed?", abort=True, default=False)
            except click.exceptions.Abort:
                click.echo("Update cancelled.")
                return

        # ── Re-ingest changed files ────────────────────────────────────────
        if changed:
            click.secho("\n--- Re-ingesting changed files ---", bold=True)
            for orig_str, corpus_key, meta, _ in changed:
                click.echo(f"\n  > Updating: {Path(orig_str).name}")
                manager.reingest_changed_file(corpus_key, meta)

        # ── Add new files ──────────────────────────────────────────────────
        if new:
            config = get_config()
            effective_type = doc_type or config.get('ingestion_config', {}).get('default_doc_type', 'document')
            click.secho(f"\n--- Adding new files (type: {effective_type}) ---", bold=True)
            manager.add_to_corpus(new, effective_type)

        click.secho("\n✅ Corpus update complete.", fg="green")

    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")


@corpus.command('ingest', hidden=True)
@click.pass_context
def ingest(ctx):
    """DEPRECATED. Use 'index build'."""
    click.secho(
        "  Warning: 'corpus ingest' is deprecated. Use 'index build' instead.",
        fg="yellow", err=True
    )
    try:
        click.echo("This will re-process the entire corpus, which can be time-consuming.")
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


@corpus.command('rebuild', hidden=True)
@click.pass_context
def rebuild(ctx):
    """DEPRECATED. Use 'index build'."""
    click.secho(
        "  Warning: 'corpus rebuild' is deprecated. Use 'index build' instead.",
        fg="yellow", err=True
    )
    ctx.invoke(ingest)


@corpus.command('summary', hidden=True)
@click.argument('identifier')
@click.pass_context
def summary(ctx, identifier):
    """DEPRECATED. Use 'index chunks <identifier> --summary'."""
    click.secho(
        "  Warning: 'corpus summary' is deprecated. Use 'index chunks <identifier> --summary' instead.",
        fg="yellow", err=True
    )
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


@corpus.command('restore')
@click.argument('identifier')
@click.option('--output', 'output_path', default=None, type=click.Path(),
              help='Write restored text to this file instead of stdout.')
@click.option('--from-store', is_flag=True, default=False,
              help='Force reassembly from index chunks (ignores text cache).')
@click.pass_context
def restore(ctx, identifier, output_path, from_store):
    """Restores plain text for a document from its text cache or the index."""
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
            click.secho(f"✅ Restored text written to: {output_path}", fg="green")
        else:
            click.echo(content)

    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")


@corpus.command('reconstitute', hidden=True)
@click.argument('identifier')
@click.option('--output', 'output_path', default=None, type=click.Path())
@click.option('--from-store', is_flag=True, default=False)
@click.pass_context
def reconstitute(ctx, identifier, output_path, from_store):
    """DEPRECATED. Use 'corpus restore'."""
    click.secho(
        "  Warning: 'corpus reconstitute' is deprecated. Use 'corpus restore' instead.",
        fg="yellow", err=True
    )
    ctx.invoke(restore, identifier=identifier, output_path=output_path, from_store=from_store)


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
