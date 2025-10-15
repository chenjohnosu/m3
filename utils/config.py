import yaml
import os

_config = None

def get_config_dir():
    """Returns the path to the ~/.monkey3 configuration directory."""
    return os.path.expanduser("~/.monkey3")

def get_config_path():
    """Returns the full path to the config.yaml file."""
    return os.path.join(get_config_dir(), "config.yaml")

def create_default_config_if_not_exists():
    """Creates a default config.yaml if it doesn't already exist."""
    config_path = get_config_path()
    if not os.path.exists(config_path):
        print("No config.yaml found. Creating a default one in ~/.monkey3/")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        default_config = {
            'project_settings': {
                'projects_directory': '~/.monkey3/projects'
            },
            'llm_providers': {
                'ollama_client': {
                    'provider': 'ollama',
                    'base_url': 'http://localhost:11434',
                    'models': {
                        'synthesis_model': {'model_name': 'llama3', 'request_timeout': 120.0},
                        'enrichment_model': {'model_name': 'mistral', 'request_timeout': 60.0}
                    }
                }
            },
            'ingestion_config': {
                'known_doc_types': [
                    'document', 'interview', 'paper', 'data',
                    'observation', 'ethnographic_notes'
                ],
                'default_doc_type': 'document',
                'cogarc_settings': {
                    'stage_0_model': 'synthesis_model',
                    'stage_1_model': 'synthesis_model',
                    'stage_2_model': 'enrichment_model',
                    'stage_3_model': 'synthesis_model'
                }
            }
        }
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, sort_keys=False)
        print(f"Default config created at: {config_path}")

def get_config():
    """Loads and returns the configuration from ~/.monkey3/config.yaml."""
    global _config
    if _config is None:
        create_default_config_if_not_exists()
        config_path = get_config_path()
        with open(config_path, 'r') as f:
            _config = yaml.safe_load(f)
    return _config