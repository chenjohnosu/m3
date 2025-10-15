import os
import json
from m3.utils.config import get_config


class ProjectManager:
    """Manages research projects, their directories, and active state."""

    def __init__(self):
        config = get_config()
        project_settings = config.get('project_settings', {})

        # All projects are now stored under the directory specified in the config
        raw_projects_dir = project_settings.get('projects_directory', '~/.monkey3/projects')
        self.projects_dir = os.path.expanduser(raw_projects_dir)

        self.active_project_file = os.path.join(os.path.dirname(self.projects_dir), '.active_project')
        os.makedirs(self.projects_dir, exist_ok=True)

    def init_project(self, project_name):
        """Initializes a new project directory."""
        project_path = os.path.join(self.projects_dir, project_name)
        if os.path.exists(project_path):
            return None, f"Project '{project_name}' already exists."

        os.makedirs(project_path)
        # You could create other subdirectories here like 'notes', 'data', etc.
        self.set_active_project(project_name)
        return project_path, f"Project '{project_name}' initialized and set as active."

    def list_projects(self):
        """Lists all available projects."""
        return [d for d in os.listdir(self.projects_dir) if os.path.isdir(os.path.join(self.projects_dir, d))]

    def set_active_project(self, project_name):
        """Sets the currently active project."""
        if not os.path.isdir(os.path.join(self.projects_dir, project_name)):
            return False, f"Project '{project_name}' not found."

        with open(self.active_project_file, 'w') as f:
            f.write(project_name)
        return True, f"'{project_name}' is now the active project."

    def get_active_project(self):
        """Gets the name and path of the currently active project."""
        if not os.path.exists(self.active_project_file):
            return None, None

        with open(self.active_project_file, 'r') as f:
            project_name = f.read().strip()

        project_path = os.path.join(self.projects_dir, project_name)
        if not os.path.isdir(project_path):
            return None, None  # Active project directory was deleted

        return project_name, project_path