from utils.config import get_prompts

class PromptManager:
    """
    Centralizes access to system prompts defined in prompts.yaml.
    """
    def __init__(self):
        self.prompts = get_prompts()

    def get(self, key, default=""):
        return self.prompts.get(key, default)
