import click
from m3.core.vector_manager import VectorManager
from m3.utils.config import get_config


@click.group()
def corpus():
    """Manages the document corpus."""
    pass


@corpus.command('add')
@click.argument('paths', nargs=-1, type=click.Path(exists=True, readable=True))
@click.option('--type', 'doc_type',
              type=click.Choice(get_config().get('ingestion_config', {}).get('known_doc_types', ['document'])),
              default=None,
              help='The type of document being added. Determines the processing pipeline.')
def add(paths, doc_type):
    """
    Adds one or more files to the corpus with a specified document type.

    This command processes the files using the appropriate ingestion pipeline
    based on the --type flag. For example, using '--type interview' will
    trigger the 'cogarc' pipeline for deep analysis.
    """
    if not paths:
        click.echo("Error: No file paths provided. Please specify one or more files to add.")
        return

    config = get_config()

    # If type is not provided, use the default from config
    if not doc_type:
        doc_type = config.get('ingestion_config', {}).get('default_doc_type', 'document')
        click.echo(f"No --type specified, using default: '{doc_type}'")

    vector_manager = VectorManager(config)
    vector_manager.add_to_corpus(list(paths), doc_type)

    click.secho(f"\n✅ Successfully added {len(paths)} file(s) to the corpus as type '{doc_type}'.", fg="green")


@corpus.command('list')
def list_files():
    """
    Lists all files currently in the corpus and their assigned type.
    """
    config = get_config()
    vector_manager = VectorManager(config)
    corpus_items = vector_manager.list_corpus()

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