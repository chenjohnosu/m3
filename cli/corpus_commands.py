import click
from core.vector_manager import VectorManager
from utils.config import get_config
from pathlib import Path


@click.group()
def corpus():
    """Manages the document corpus for the active project."""
    pass


@corpus.command('add')
@click.argument('paths', nargs=-1, type=click.Path(exists=True, readable=True))
@click.option('--type', 'doc_type',
              type=click.Choice(get_config().get('ingestion_config', {}).get('known_doc_types', ['document'])),
              default=None,
              help='The type of document being added.')
def add(paths, doc_type):
    """Adds one or more files/directories to the active project's corpus."""
    if not paths:
        click.echo("Error: No file paths provided.")
        return

    config = get_config()
    if not doc_type:
        doc_type = config.get('ingestion_config', {}).get('default_doc_type', 'document')
        click.echo(f"No --type specified, using default: '{doc_type}'")

    try:
        manager = VectorManager(config)
        manager.add_to_corpus(list(paths), doc_type)
        click.secho(f"\n✅ Successfully added and processed {len(paths)} path(s).", fg="green")
    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")


@corpus.command('remove')
@click.argument('identifier')
def remove(identifier):
    """Removes a file from the corpus by its original filename or ID."""
    try:
        manager = VectorManager()
        success, message = manager.remove_from_corpus(identifier)
        if success:
            click.secho(f"Success: {message}", fg="green")
        else:
            click.secho(f"Error: {message}", fg="red")
    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")


@corpus.command('list')
def list_files():
    """Lists all files in the active project's corpus."""
    try:
        manager = VectorManager()
        corpus_items = manager.list_corpus()
        if not corpus_items:
            click.echo("The corpus is currently empty.")
            return

        click.echo("\n--- Corpus Contents ---")
        click.echo(f"{'ID':<38} | {'Chunks':<8} | {'Document Type':<20} | {'Original File'}")
        click.echo("-" * 100)

        # Sort items by document type, then by original filename
        sorted_items = sorted(
            corpus_items.items(),
            key=lambda item: (item[1].get('doc_type', 'N/A'), Path(item[1].get('original_path', '')).name)
        )

        for path_in_corpus, meta in sorted_items:
            # The ID is the UUID-based name of the file stored in the corpus
            file_id = Path(path_in_corpus).stem
            doc_type = meta.get('doc_type', 'N/A')
            original_filename = Path(meta.get('original_path', 'Unknown')).name

            # Get the chunk count for the current document
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
        # Add a confirmation prompt that defaults to 'No' for safety.
        click.confirm("Are you sure you want to proceed?", abort=True, default=False)
        manager = VectorManager()
        manager.rebuild_vector_store()
    except click.exceptions.Abort:
        click.echo("Operation cancelled by user.")
    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")


@corpus.command('rebuild')
@click.pass_context
def rebuild(ctx):
    """Alias for 'ingest'. Rebuilds the entire vector store."""
    # This invokes the 'ingest' command, ensuring the logic is not duplicated.
    ctx.invoke(ingest)