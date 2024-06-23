from llm_tongyi import TongyiLLM
from http import HTTPStatus

tongyi = TongyiLLM()
resp = tongyi.chat("你好，你能做什么")

if resp.status_code == HTTPStatus.OK:
    print(resp.output.text, end="")  # 输出文本
else:
    print(resp.code)  # 错误码
    print(resp.message)  # 错误信息