import yaml
from pathlib import Path

# The default base directory is now managed within the config utility
M3_BASE_DIR = Path.home() / ".monkey3"
CONFIG_PATH = M3_BASE_DIR / "config.yaml"

def load_config():
    """Loads the application configuration from a YAML file."""
    if not CONFIG_PATH.exists():
        # Create a default configuration if it doesn't exist
        default_config = {
            "base_dir": str(M3_BASE_DIR),
            "llm_model": "llama3",
            "embed_model": "mxbai-embed-large",
            "active_project": None
        }
        save_config(default_config)
        return default_config

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
        if 'active_project' not in config:
            config['active_project'] = None
        return config

def save_config(config):
    """Saves the application configuration to a YAML file."""
    # Ensure the base directory exists before saving the config
    M3_BASE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def set_active_project(project_name):
    """Sets the active project in the config."""
    config = load_config()
    config['active_project'] = project_name
    save_config(config)

def get_active_project():
    """Gets the active project from the config."""
    config = load_config()
    return config.get('active_project')