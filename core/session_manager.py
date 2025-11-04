import click
from utils.config import get_config
from core.project_manager import ProjectManager
from core.llm_manager import LLMManager
from core.plugin_manager import PluginManager
from core.vector_manager import VectorManager
from core.analyze_manager import AnalyzeManager

class M3Session:
    """
    Manages the persistent state for an M3 interactive session.
    This object is created once and passed into the click context (ctx.obj).
    """
    def __init__(self):
        click.echo("Initializing M3 session...")
        try:
            self.config = get_config()
            self.project_manager = ProjectManager()
            self.llm_manager = LLMManager(self.config)
            self.plugin_manager = PluginManager()

            # --- Project-specific ---
            self.active_project_name = None
            self.active_project_path = None
            self.vector_manager = None
            self.analyze_manager = None

            # Load the active project on startup
            active_name, _ = self.project_manager.get_active_project()
            if active_name:
                self.load_project(active_name)

            click.echo("Session ready.")

        except Exception as e:
            click.secho(f"🔥 Fatal Error during session startup: {e}", fg="red")
            click.secho("  > Please check your config.yaml and ensure dependencies are installed.", fg="yellow")
            click.secho("  > Exiting.", fg="red")
            exit(1)

    def load_project(self, project_name):
        """
        Loads a new project into the session, re-instantiating
        the necessary project-specific managers.
        """
        if project_name is None:
            self.active_project_name = None
            self.active_project_path = None
            self.vector_manager = None
            self.analyze_manager = None
            click.echo("Active project cleared.")
            return

        project_path = self.project_manager.get_project_path_by_name(project_name)
        if not project_path:
            click.secho(f"Error: Could not load project '{project_name}'.", fg="red")
            return

        try:
            click.echo(f"Loading project '{project_name}' into session...")
            self.active_project_name = project_name
            self.active_project_path = project_path

            # Instantiate project-specific managers
            self.vector_manager = VectorManager(
                self.config,
                self.active_project_name,
                self.active_project_path,
                self.llm_manager
            )

            self.analyze_manager = AnalyzeManager(
                self.config,
                self.active_project_name,
                self.active_project_path,
                self.llm_manager,
                self.plugin_manager
            )

            click.echo(f"Successfully loaded '{project_name}'.")

        except Exception as e:
            click.secho(f"🔥 Error loading project '{project_name}': {e}", fg="red")
            self.active_project_name = None
            self.active_project_path = None
            self.vector_manager = None
            self.analyze_manager = None

    def get_project_prompt(self):
        """Returns the prompt string for the REPL."""
        return f"[m3:{self.active_project_name}]> " if self.active_project_name else "[m3]> "