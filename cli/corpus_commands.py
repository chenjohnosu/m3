import click
from ..core import project_manager

@click.group()
def corpus():
    """Commands for grouping and managing collections of projects."""
    pass

@corpus.command('create')
@click.argument('corpus_name')
def create(corpus_name):
    """Creates a new corpus."""
    project_manager.create_corpus(corpus_name)

@corpus.command('list')
def list_corpora():
    """Lists all available corpora."""
    project_manager.list_corpora()

@corpus.command('add-project')
@click.argument('corpus_name')
@click.option('-p', '--project-name', required=True, help='Name of the project to add.')
def add_project(corpus_name, project_name):
    """Adds an existing project to the specified corpus."""
    project_manager.add_project_to_corpus(corpus_name, project_name)

@corpus.command('remove-project')
@click.argument('corpus_name')
@click.option('-p', '--project-name', required=True, help='Name of the project to remove.')
def remove_project(corpus_name, project_name):
    """Removes a project from a corpus."""
    project_manager.remove_project_from_corpus(corpus_name, project_name)

@corpus.command('delete')
@click.argument('corpus_name')
def delete(corpus_name):
    """Deletes a corpus definition."""
    project_manager.delete_corpus(corpus_name)
