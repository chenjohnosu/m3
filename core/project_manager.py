import click
import shutil
from pathlib import Path
from utils.config import load_config, set_active_project, get_active_project

# Load the configuration at the module level
config = load_config()

# The base directory is now read from the config file
M3_BASE_DIR = Path(config.get("base_dir", Path.home() / ".monkey3"))
PROJECTS_DIR = M3_BASE_DIR / "projects"

# --- Project Management ---

def create_project(project_name):
    """Core logic to create a project."""
    click.echo("==> Task: Create project" + project_name)
    project_path = PROJECTS_DIR / project_name
    if not project_path.exists():
        project_path = PROJECTS_DIR / project_name
        corpus_path = project_path / "corpus"
        vector_store_path = project_path / "vector_store"

    if project_path.exists():
        click.echo(f"Error: Project '{project_name}' already exists at {project_path}")
        return

    try:
        # Create the directory hierarchy
        corpus_path.mkdir(parents=True, exist_ok=True)
        vector_store_path.mkdir(parents=True, exist_ok=True)
        click.echo(f"Project '{project_name}' created successfully.")
        click.echo(f"  - Corpus directory: {corpus_path}")
        click.echo(f"  - Vector store: {vector_store_path}")
    except OSError as e:
        click.echo(f"Error creating project '{project_name}': {e}", err=True)

def list_projects():
    """Placeholder for listing all projects."""
    click.echo("==> Task: List all projects")
    """Core logic to list all projects."""
    if not PROJECTS_DIR.is_dir():
        click.echo("No projects found. The projects directory does not exist.")
        return

    projects = [p.name for p in PROJECTS_DIR.iterdir() if p.is_dir()]
    active_project = get_active_project()

    if not projects:
        click.echo("No projects found.")
    else:
        click.echo("Available projects:")
        for project in sorted(projects):
            if project == active_project:
                click.echo(f"- {project} (active)")
            else:
                click.echo(f"- {project}")

def set_active(project_name):
    """Sets the active project."""
    project_path = PROJECTS_DIR / project_name
    if not project_path.exists():
        click.echo(f"Error: Project '{project_name}' not found.")
        return
    set_active_project(project_name)
    click.echo(f"Project '{project_name}' is now active.")


def add_file_to_project(project_name, file_path):
    """Placeholder for adding a file to a project."""
    click.echo(f"==> Task: Add file '{file_path}' to project '{project_name}'")

def remove_file_from_project(project_name, file_name):
    """Placeholder for removing a file from a project."""
    click.echo(f"==> Task: Remove file '{file_name}' from project '{project_name}'")

def get_project_status(project_name):
    """Placeholder for getting a project's status."""
    click.echo(f"==> Task: Get status for project '{project_name}'")

def delete_project(project_name):
    """Core logic to remove a project."""
    project_path = PROJECTS_DIR / project_name

    if not project_path.exists():
        click.echo(f"Error: Project '{project_name}' not found.")
        return

    # Prompt for confirmation
    if click.confirm(f"Are you sure you want to permanently remove the project '{project_name}' and all its data?", default=False):
        try:
            shutil.rmtree(project_path)
            click.echo(f"Project '{project_name}' has been removed.")
        except OSError as e:
            click.echo(f"Error removing project '{project_name}': {e}", err=True)
    else:
        click.echo("Operation cancelled.")


# --- Corpus Management ---

def create_corpus(corpus_name):
    """Placeholder for creating a corpus."""
    click.echo(f"==> Task: Create corpus '{corpus_name}'")

def list_corpora():
    """Placeholder for listing all corpora."""
    click.echo("==> Task: List all corpora")

def add_project_to_corpus(corpus_name, project_name):
    """Placeholder for adding a project to a corpus."""
    click.echo(f"==> Task: Add project '{project_name}' to corpus '{corpus_name}'")

def remove_project_from_corpus(corpus_name, project_name):
    """Placeholder for removing a project from a corpus."""
    click.echo(f"==> Task: Remove project '{project_name}' from corpus '{corpus_name}'")

def delete_corpus(corpus_name):
    """Placeholder for deleting a corpus."""
    click.echo(f"==> Task: Delete corpus '{corpus_name}'")