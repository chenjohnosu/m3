import click

def get_status(project_name):
    """Core logic to get vector store status."""
    click.echo(f"CORE: Getting vector store status for project '{project_name}'...")

def rebuild(project_name):
    """Core logic to rebuild a vector store."""
    click.echo(f"CORE: Rebuilding vector store for project '{project_name}'...")
