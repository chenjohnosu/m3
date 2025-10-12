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
            "embed_model": "mxbai-embed-large"
        }
        save_config(default_config)
        return default_config

    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def save_config(config):
    """Saves the application configuration to a YAML file."""
    # Ensure the base directory exists before saving the config
    M3_BASE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)