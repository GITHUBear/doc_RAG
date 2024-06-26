import os
from zhipuai import ZhipuAI
from dotenv import load_dotenv
from config import RagConfig
from llm_abc import LLM


class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def __str__(self):
        return f"{self.role}: {self.content}"

    def to_dict(self):
        return {"role": self.role, "content": self.content}


def default_message(input_prompt: str):
    return [
        {
            "role": "user",
            "content": input_prompt,
        }
    ]


class ZhipuLLM(LLM):
    def __init__(self) -> None:
        load_dotenv(".env")
        config = RagConfig()
        self.model_name = config.zhipu_model_name
        self.top_p = config.zhipu_top_p
        self.temperature = config.zhipu_temperature
        self.stream = config.zhipu_stream
        self._client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))

    def chat(self, input_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=default_message(input_prompt),
            stream=self.stream,
            top_p=self.top_p,
            temperature=self.temperature,
        )
        return response.choices[0].message.content
