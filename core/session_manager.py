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
        click.echo("Initializing M3 session components...")
        try:
            self.config = get_config()
            self.project_manager = ProjectManager()
            self.llm_manager = LLMManager(self.config)

            click.echo("Loading plugins...", nl=False)
            self.plugin_manager = PluginManager()
            click.echo(" Done.")

            # --- Project-specific (Lazy Loaded) ---
            self.active_project_name = None
            self.active_project_path = None

            # Internal storage for lazy properties
            self._vector_manager = None
            self._analyze_manager = None

            # Load the active project info (fast)
            active_name, _ = self.project_manager.get_active_project()
            if active_name:
                self.load_project(active_name)

            click.echo("Session ready.")

        except Exception as e:
            click.secho(f"🔥 Fatal Error during session startup: {e}", fg="red")
            click.secho("  > Please check your config.yaml and ensure dependencies are installed.", fg="yellow")
            click.secho("  > Exiting.", fg="red")
            exit(1)

    @property
    def vector_manager(self):
        """Lazy loader for VectorManager."""
        if self._vector_manager is None and self.active_project_name:
            click.secho(f"  > Initializing VectorManager for '{self.active_project_name}'...", dim=True)
            try:
                self._vector_manager = VectorManager(
                    self.config,
                    self.active_project_name,
                    self.active_project_path,
                    self.llm_manager
                )
            except Exception as e:
                click.secho(f"🔥 Error initializing VectorManager: {e}", fg="red")
                return None
        return self._vector_manager

    @property
    def analyze_manager(self):
        """Lazy loader for AnalyzeManager."""
        if self._analyze_manager is None and self.active_project_name:
            click.secho(f"  > Initializing AnalyzeManager for '{self.active_project_name}'...", dim=True)
            try:
                self._analyze_manager = AnalyzeManager(
                    self.config,
                    self.active_project_name,
                    self.active_project_path,
                    self.llm_manager,
                    self.plugin_manager
                )
            except Exception as e:
                click.secho(f"🔥 Error initializing AnalyzeManager: {e}", fg="red")
                return None
        return self._analyze_manager

    def load_project(self, project_name):
        """
        Sets the active project path but does NOT instantiate managers immediately.
        They will be loaded on first access via properties.
        """
        if project_name is None:
            self.active_project_name = None
            self.active_project_path = None
            self._vector_manager = None
            self._analyze_manager = None
            click.echo("Active project cleared.")
            return

        project_path = self.project_manager.get_project_path_by_name(project_name)
        if not project_path:
            click.secho(f"Error: Could not load project '{project_name}'.", fg="red")
            return

        # Just set the paths and clear the cached managers
        self.active_project_name = project_name
        self.active_project_path = project_path
        self._vector_manager = None
        self._analyze_manager = None

        click.echo(f"Active project set to: '{project_name}'")

    def get_project_prompt(self):
        """Returns the prompt string for the REPL."""
        return f"[m3:{self.active_project_name}]> " if self.active_project_name else "[m3]> "

    def get_styled_prompt(self):
        """Returns prompt_toolkit FormattedText for the styled prompt."""
        from prompt_toolkit.formatted_text import FormattedText
        project = self.active_project_name or 'm3'
        return FormattedText([
            ('class:bracket', '['),
            ('class:appname', 'm3'),
            ('class:colon',   ':'),
            ('class:project', project),
            ('class:bracket', ']'),
            ('class:arrow',   '> '),
        ])