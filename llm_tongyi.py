import os
import dashscope
from dotenv import load_dotenv
from config import RagConfig

class TongyiLLM:
    def __init__(self) -> None:
        load_dotenv(".env")
        config = RagConfig()
        self.model_name = config.tongyi_model_name
        self.top_p = config.tongyi_top_p
        self.temperature = config.tongyi_temperature
        self.stream = config.tongyi_stream

    def chat(self, input_prompt):
        response = dashscope.Generation.call(
            model=self.model_name,
            prompt=input_prompt,
            history=None,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            stream=self.stream,
            top_p=self.top_p,
            temperature=self.temperature,
        )
        return response