from abc import ABC, abstractmethod

class BasePipeline(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def run(self, documents, doc_type):
        pass