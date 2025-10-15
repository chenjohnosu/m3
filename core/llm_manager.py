from llama_index.llms.ollama import Ollama


class LLMManager:
    """
    Manages the creation and retrieval of LLM instances based on the
    configuration in config.yaml.
    """

    def __init__(self, config):
        self.config = config
        self.clients = {}
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
        """
        for client_name, client_data in self.clients.items():
            if model_key in client_data['models']:
                model_info = client_data['models'][model_key]
                print(f"  > Instantiating LLM '{model_info['model_name']}' for role '{model_key}'...")
                return Ollama(
                    model=model_info['model_name'],
                    base_url=client_data['base_url'],
                    request_timeout=model_info.get('request_timeout', 60.0)
                )

        raise ValueError(f"LLM model key '{model_key}' not found in llm_providers in config.yaml")