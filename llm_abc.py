from abc import ABC, abstractmethod


class LLM(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def chat(self, input_prompt: str) -> str:
        pass
