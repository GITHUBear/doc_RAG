import os
import dashscope
from dotenv import load_dotenv
from config import RagConfig
from typing import List
from http import HTTPStatus

class TongyiLLM:
    def __init__(self) -> None:
        load_dotenv(".env")
        config = RagConfig()
        self.model_name = config.tongyi_model_name
        self.top_p = config.tongyi_top_p
        self.temperature = config.tongyi_temperature
        self.stream = config.tongyi_stream
        self.multi_chat_max_msgs = config.multi_chat_max_rounds * 2

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
    
    def multi_chat(self, messages: List[dict], user_content, pure_user_content):
        messages = messages[2 - self.multi_chat_max_msgs:]
        new_msgs = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
        new_msgs.extend(messages)
        new_msgs.append({
            'role': 'user',
            'content': user_content,
        })
        response = dashscope.Generation.call(
            model=self.model_name,
            messages=new_msgs,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            stream=self.stream,
            top_p=self.top_p,
            temperature=self.temperature,
            result_format='message'
        )
        if response.status_code == HTTPStatus.OK:
            messages.append({
                'role': 'user',
                'content': pure_user_content,
            })
            messages.append({
                'role': response.output.choices[0]['message']['role'],
                'content': response.output.choices[0]['message']['content']
            })
            return response.status_code, response.output.choices[0]['message']['content'], messages
        else:
            return response.status_code, "", messages