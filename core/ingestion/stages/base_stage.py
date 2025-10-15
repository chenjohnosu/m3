from abc import ABC, abstractmethod

class BaseStage(ABC):
    def __init__(self, config, llm=None):
        self.config = config
        self.llm = llm

    @abstractmethod
    def process(self, data):
        pass