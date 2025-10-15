import click
from core.vector_manager import VectorManager
from utils.config import get_config

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
@click.argument('filename')
def remove(filename):
    """Removes a file from the corpus and vector store."""
    try:
        manager = VectorManager()
        success, message = manager.remove_from_corpus(filename)
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
        click.echo(f"{'Document Type':<20} | {'File Path'}")
        click.echo("-" * 70)
        for path, meta in sorted(corpus_items.items(), key=lambda item: item[1]['doc_type']):
            doc_type = meta.get('doc_type', 'N/A')
            click.secho(f"{doc_type:<20}", fg="cyan", nl=False)
            click.echo(f" | {path}")
        click.echo("-" * 70)
    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")

@corpus.command('ingest')
def ingest():
    """Processes all new or updated files in the corpus."""
    try:
        click.echo("This command will re-process the entire corpus, which can be time-consuming.")
        click.confirm("Are you sure you want to proceed?", abort=True)
        manager = VectorManager()
        manager.rebuild_vector_store()
    except Exception as e:
        click.secho(f"🔥 Error: {e}", fg="red")