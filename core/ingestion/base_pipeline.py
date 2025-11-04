from abc import ABC, abstractmethod

class BasePipeline(ABC):
    def __init__(self, config, llm_manager=None):
        self.config = config
        self.llm_manager = llm_manager

    @abstractmethod
    def run(self, documents, doc_type):
        pass