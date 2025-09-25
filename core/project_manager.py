import click

def create(project_name):
    """Core logic to create a project."""
    click.echo(f"CORE: Creating project '{project_name}'...")

def list_all():
    """Core logic to list all projects."""
    click.echo("CORE: Listing all projects...")

def remove(project_name):
    """Core logic to remove a project."""
    click.echo(f"CORE: Removing project '{project_name}'...")

def add_to_corpus(project_name, file_path):
    """Core logic to add a file to a corpus."""
    click.echo(f"CORE: Adding file '{file_path}' to project '{project_name}' corpus...")

def remove_from_corpus(project_name, filename):
    """Core logic to remove a file from a corpus."""
    click.echo(f"CORE: Removing file '{filename}' from project '{project_name}' corpus...")

def list_corpus(project_name):
    """Core logic to list files in a corpus."""
    click.echo(f"CORE: Listing corpus for project '{project_name}'...")
