from abc import ABC, abstractmethod
from typing import List, Generator, Any


class LLM(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def chat(self, input_prompt: str) -> str:
        pass

    @abstractmethod
    def chat_with_history(
        self, messages: List[dict], stream=False, need_append=False
    ) -> str | Generator[bytes, Any, None]:
        """
        Chat with history.
        
        Args:
            messages (List[dict]): List of messages. [{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'hi'}]
            stream (bool, optional): Stream or not. Defaults to False.
            need_append (bool, optional): Need append or not. Defaults to True.
        """
        pass