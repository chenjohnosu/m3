from llama_index.llms.ollama import Ollama
from utils.config import get_config
import click


class LLMManager:
    """
    Manages the creation and retrieval of LLM instances based on the
    configuration in config.yaml.
    """

    def __init__(self, config):
        self.config = config
        self.clients = {}
        self.loaded_models = {}  # <-- ADDED: Cache for loaded models
        self._load_clients()

    def _load_clients(self):
        """Initializes LLM clients from the config."""
        providers = self.config.get('llm_providers', {})
        for client_name, client_config in providers.items():
            if client_config.get('provider') == 'ollama':
                self.clients[client_name] = {
                    'base_url': client_config.get('base_url'),
                    'models': client_config.get('models', {})
                }

    def get_llm(self, model_key):
        """
        Gets a LlamaIndex LLM instance for a given model key (e.g., 'synthesis_model').
        Caches the instance after first creation.
        """

        # --- 1. Check if model is already cached ---
        if model_key in self.loaded_models:
            # Use err=True to ensure this logs outside of a --quiet flag
            click.echo(f"  > Using cached LLM for role '{model_key}'...", err=True)
            return self.loaded_models[model_key]
        # --- END CACHE CHECK ---

        ingestion_conf = self.config.get('ingestion_config', {})
        cogarc_settings = ingestion_conf.get('cogarc_settings', {})

        # Determine which model key to use
        if model_key == 'synthesis_model':
            model_key_to_use = cogarc_settings.get('stage_3_model')
        elif model_key == 'enrichment_model':
            model_key_to_use = cogarc_settings.get('stage_2_model')
        elif model_key == 'stratify_model':
            model_key_to_use = cogarc_settings.get('stage_0_model')
        elif model_key == 'structure_model':
            model_key_to_use = cogarc_settings.get('stage_1_model')
        else:
            # Fallback for other keys like 'default_model'
            model_key_to_use = model_key

        if not model_key_to_use:
            raise ValueError(f"No model key defined for role '{model_key}' in cogarc_settings.")

        for client_name, client_data in self.clients.items():
            if model_key_to_use in client_data['models']:
                model_info = client_data['models'][model_key_to_use]

                # --- MODIFIED: Print only when instantiating ---
                # Use err=True to ensure this logs outside of a --quiet flag
                click.echo(
                    f"  > Instantiating LLM '{model_info['model_name']}' for role '{model_key}' (using key '{model_key_to_use}')...",
                    err=True)

                # --- 2. Create the new LLM instance ---
                llm_instance = Ollama(
                    model=model_info['model_name'],
                    base_url=client_data['base_url'],
                    request_timeout=model_info.get('request_timeout', 120.0)
                )

                # --- 3. Store the new instance in the cache ---
                self.loaded_models[model_key] = llm_instance

                # --- 4. Return the new instance ---
                return llm_instance

        raise ValueError(
            f"LLM model key '{model_key_to_use}' (for role '{model_key}') not found in llm_providers in config.yaml")