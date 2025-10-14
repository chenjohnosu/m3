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
            "llm_model": "llama3",  # This is for future LLM chat features

            "embedding_profiles": {
                "default_multilingual": {
                    "embed_model": "intfloat/multilingual-e5-large",
                    "chunk_size": 512,
                    "chunk_overlap": 50
                },
                "english_fast": {
                    "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "chunk_size": 256,
                    "chunk_overlap": 20
                }
            },

            "active_profile": "default_multilingual",
            "active_project": None
        }
        save_config(default_config)
        return default_config

    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


def save_config(config):
    """Saves the application configuration to a YAML file."""
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