import requests

# API endpoint
url = 'http://127.0.0.1:5000/chat'  # 如果API不是在localhost运行，则替换为相应的主机和端口

# 提出你的问题
question = '你好，我有个问题。'

# 构造请求的数据
data = {
    'question': question
}

# 发送请求
response = requests.post(url, json=data)

# 检查响应状态码
if response.status_code == 200:
    # 解析响应数据
    data = response.json()
    print('回答:', data['answer'])
else:
    print('请求失败，状态码:', response.status_code)
