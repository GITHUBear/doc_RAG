from llm_tongyi import TongyiLLM
from http import HTTPStatus
import time

tongyi = TongyiLLM()
start_time = time.time()
resp = tongyi.chat("OceanBase如何查询事务隔离级别")
end_time = time.time()
print(f"cost: {(end_time - start_time) * 1000}ms")

if resp.status_code == HTTPStatus.OK:
    print(resp.output.text, end="")  # 输出文本
else:
    print(resp.code)  # 错误码
    print(resp.message)  # 错误信息