import os
import dashscope
from dotenv import load_dotenv
from config import RagConfig
from typing import List
from llm_abc import LLM
from typing import List, Generator, Any
from message import StreamChunk
from openai import OpenAI, Stream


class TongyiLLM(LLM):
    __endpoint = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(self) -> None:
        load_dotenv(".env")
        config = RagConfig()
        self.model_name = config.tongyi_model_name
        self.top_p = config.tongyi_top_p
        self.temperature = config.tongyi_temperature
        self.stream = config.tongyi_stream
        self.multi_chat_max_msgs = config.multi_chat_max_rounds * 2

        self._client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=self.__endpoint,
        )

    def chat(self, input_prompt: str, stream=False) -> str:
        response = dashscope.Generation.call(
            model=self.model_name,
            prompt=input_prompt,
            history=None,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            stream=self.stream,
            top_p=self.top_p,
            temperature=self.temperature,
        )
        if response.status_code == 200:
            return response.output.text
        else:
            print(response.code, response.message)
            raise ValueError(f"{response.code}: {response.message}")

    def chat_with_history(
        self, messages: List[dict], stream=False, need_append=False
    ) -> str | Generator[bytes, Any, None]:
        messages = messages[-self.multi_chat_max_msgs :]
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=stream,
            top_p=self.top_p,
            temperature=self.temperature,
        )
        if not stream:
            return response.choices[0]["message"]["content"]

        def response_stream():
            for chunk in response:
                content = chunk.choices[0].delta.content
                model = self.model_name
                stream_chunk = StreamChunk(
                    model=model,
                    content=content,
                )
                yield stream_chunk.to_json().encode()

            if not need_append:
                yield StreamChunk(
                    model=self.model_name, content="", done=True
                ).to_json().encode()

        return response_stream()

    def multi_chat(self, messages: List[dict], user_content, pure_user_content):
        messages = messages[2 - self.multi_chat_max_msgs :]
        new_msgs = [{"role": "system", "content": "You are a helpful assistant."}]
        new_msgs.extend(messages)
        new_msgs.append(
            {
                "role": "user",
                "content": user_content,
            }
        )
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=new_msgs,
            stream=self.stream,
            top_p=self.top_p,
            temperature=self.temperature,
        )

        response_content = response.choices[0]["message"]["content"]

        messages.append(
            {
                "role": "user",
                "content": pure_user_content,
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": response_content,
            }
        )
        return (
            response.status_code,
            response_content,
            messages,
        )
