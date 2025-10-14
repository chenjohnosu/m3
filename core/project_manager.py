import click
import shutil
import glob
import json
import uuid
import hashlib
from pathlib import Path
from utils.config import load_config, set_active_project, get_active_project
from core import vector_manager

config = load_config()
M3_BASE_DIR = Path(config.get("base_dir", Path.home() / ".monkey3"))
PROJECTS_DIR = M3_BASE_DIR / "projects"
MANIFEST_NAME = "corpus_manifest.json"


# --- Helper Functions ---

def _load_manifest(corpus_path):
    manifest_path = corpus_path / MANIFEST_NAME
    if not manifest_path.exists():
        return {}
    with open(manifest_path, 'r') as f:
        return json.load(f)


def _save_manifest(corpus_path, manifest_data):
    manifest_path = corpus_path / MANIFEST_NAME
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f, indent=2)


def _calculate_hash(file_path):
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# --- Project Management ---

def create_project(project_name):
    project_path = PROJECTS_DIR / project_name
    corpus_path = project_path / "corpus"
    if project_path.exists():
        click.echo(f"Error: Project '{project_name}' already exists.")
        return
    click.echo(f"Creating project '{project_name}'...")
    try:
        corpus_path.mkdir(parents=True, exist_ok=True)
        _save_manifest(corpus_path, {})
        click.echo(f"  -> Project directories created successfully.")
        vector_manager.create_store(project_name)
        click.echo(f"\nProject '{project_name}' is ready.")
    except OSError as e:
        click.echo(f"Error creating project '{project_name}': {e}", err=True)


def list_projects():
    if not PROJECTS_DIR.is_dir():
        click.echo("No projects found.")
        return
    projects = [p.name for p in PROJECTS_DIR.iterdir() if p.is_dir()]
    active_project = get_active_project()
    if not projects:
        click.echo("No projects found.")
    else:
        click.echo("Available projects:")
        for p in sorted(projects):
            if p == active_project:
                click.echo(f"- {p} (active)")
            else:
                click.echo(f"- {p}")


def set_active(project_name):
    project_path = PROJECTS_DIR / project_name
    if not project_path.exists():
        click.echo(f"Error: Project '{project_name}' not found.")
        return
    set_active_project(project_name)
    click.echo(f"Project '{project_name}' is now active.")


def delete_project(project_name):
    project_path = PROJECTS_DIR / project_name
    if not project_path.exists():
        click.echo(f"Error: Project '{project_name}' not found.")
        return
    if click.confirm(f"Are you sure you want to permanently remove '{project_name}'?", default=False):
        try:
            shutil.rmtree(project_path)
            click.echo(f"Project '{project_name}' has been removed.")
        except OSError as e:
            click.echo(f"Error removing project '{project_name}': {e}", err=True)
    else:
        click.echo("Operation cancelled.")


# --- Corpus Management for a Project ---

def add_to_corpus(project_name, paths):
    project_path = PROJECTS_DIR / project_name
    corpus_path = project_path / "corpus"
    if not project_path.exists():
        click.echo(f"Error: Project '{project_name}' not found.")
        return
    manifest = _load_manifest(corpus_path)
    files_to_process = set()
    for path_str in paths:
        expanded_paths = glob.glob(path_str, recursive=True)
        if not expanded_paths:
            click.echo(f"Warning: No files matching '{path_str}'.")
            continue
        for p in expanded_paths:
            path = Path(p)
            if path.is_file():
                files_to_process.add(path)
    if not files_to_process:
        click.echo("No valid files found to add.")
        return
    added_count = 0
    updated_count = 0
    for file_path in files_to_process:
        original_filename = file_path.name
        new_hash = _calculate_hash(file_path)
        if original_filename in manifest:
            entry = manifest[original_filename]
            if entry['content_hash'] == new_hash:
                click.echo(f"  -> Skipped '{original_filename}' (unchanged).")
                continue
            entry['content_hash'] = new_hash
            entry['version'] += 1
            internal_path = corpus_path / f"{entry['uuid']}{file_path.suffix}"
            shutil.copy2(file_path, internal_path)
            click.echo(f"  -> Updated '{original_filename}' (version {entry['version']}).")
            updated_count += 1
        else:
            file_uuid = str(uuid.uuid4())
            internal_path = corpus_path / f"{file_uuid}{file_path.suffix}"
            manifest[original_filename] = {
                'uuid': file_uuid,
                'content_hash': new_hash,
                'version': 1
            }
            shutil.copy2(file_path, internal_path)
            click.echo(f"  -> Added '{original_filename}'.")
            added_count += 1
    _save_manifest(corpus_path, manifest)
    click.echo(f"\nProcess complete. Added: {added_count}, Updated: {updated_count}.")


def list_corpus(project_name):
    project_path = PROJECTS_DIR / project_name
    corpus_path = project_path / "corpus"
    if not project_path.exists():
        click.echo(f"Error: Project '{project_name}' not found.", err=True)
        return
    manifest = _load_manifest(corpus_path)
    if not manifest:
        click.echo(f"The corpus for project '{project_name}' is empty.")
        return
    click.echo(f"Corpus for project '{project_name}':")
    table_data = []
    for filename, data in manifest.items():
        table_data.append([filename, f"v{data['version']}", data['uuid']])
    col_widths = [max(len(str(item)) for item in col) for col in zip(*table_data)]
    header = f"{'FILENAME'.ljust(col_widths[0])}  {'VERSION'.ljust(col_widths[1])}  {'UUID'.ljust(col_widths[2])}"
    click.echo(header)
    click.echo(f"{'-' * col_widths[0]}  {'-' * col_widths[1]}  {'-' * col_widths[2]}")
    for row in table_data:
        click.echo(f"{row[0].ljust(col_widths[0])}  {row[1].ljust(col_widths[1])}  {row[2].ljust(col_widths[2])}")


def remove_from_corpus(project_name, filename):
    """Core logic to remove a file from a project's corpus and vector store."""
    project_path = PROJECTS_DIR / project_name
    corpus_path = project_path / "corpus"
    if not project_path.exists():
        click.echo(f"Error: Project '{project_name}' not found.", err=True)
        return

    manifest = _load_manifest(corpus_path)
    if filename not in manifest:
        click.echo(f"Error: File '{filename}' not found in the corpus.", err=True)
        return

    # Get file info before deleting from manifest
    entry = manifest[filename]
    doc_uuid = entry['uuid']
    original_suffix = Path(filename).suffix
    internal_filename = f"{doc_uuid}{original_suffix}"
    internal_filepath = corpus_path / internal_filename

    # 1. Remove from file system
    if internal_filepath.exists():
        try:
            internal_filepath.unlink()
            click.echo(f"  -> Removed '{filename}' from corpus directory.")
        except OSError as e:
            click.echo(f"Error removing file from disk: {e}", err=True)
            return
    else:
        click.echo(f"Warning: Internal file '{internal_filename}' not found, but proceeding.")

    # 2. Remove from manifest
    del manifest[filename]
    _save_manifest(corpus_path, manifest)

    # 3. Remove from vector store
    vector_manager.remove_document(project_name, doc_uuid)

    click.echo(f"Successfully removed '{filename}'.")