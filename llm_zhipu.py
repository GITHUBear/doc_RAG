import os
from zhipuai import ZhipuAI
from dotenv import load_dotenv
from config import RagConfig
from llm_abc import LLM
from typing import List, Generator, Any
from message import StreamChunk


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
        self.multi_chat_max_msgs = config.multi_chat_max_rounds * 2
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

    def chat_with_history(
        self,
        messages: List[dict],
        stream=False,
        need_append=False,
    ) -> str | Generator[bytes, Any, None]:
        messages = messages[-self.multi_chat_max_msgs:]
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=stream,
            top_p=self.top_p,
            temperature=self.temperature,
        )
        if not stream:
            return response.choices[0].message.content

        def response_stream():
            for chunk in response:
                # choices[0].delta in Zhipu
                content = chunk.choices[0].delta.content
                stream_chunk = StreamChunk(
                    content=content,
                    model=self.model_name,
                )
                yield stream_chunk.to_json().encode()

            if not need_append:
                yield StreamChunk(model=self.model_name, content="", done=True).to_json().encode()

        return response_stream()
