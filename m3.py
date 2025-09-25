import click

from .cli.project_commands import project
from .cli.corpus_commands import corpus
from .cli.vector_commands import vector

@click.group()
def cli():
    """
    monkey3 (m3): A tool for qualitative data analysis powered by local LLMs.
    """
    pass

# Add the command groups to the main CLI tool
cli.add_command(project)
cli.add_command(corpus)
cli.add_command(vector)

if __name__ == '__main__':
    cli()
