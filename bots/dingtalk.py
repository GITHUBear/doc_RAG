from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
import os
import threading
from typing import Union, List
from typing_extensions import Annotated

app = FastAPI()


class TextMsg(BaseModel):
    content: str


class RichTextMsg(BaseModel):
    text: Union[str, None] = None
    downloadCode: Union[str, None] = None
    type: Union[str, None] = None


class ContentMsg(BaseModel):
    richText: Union[List[RichTextMsg], None] = None
    downloadCode: Union[str, None] = None


class QueryRequest(BaseModel):
    conversationId: str
    msgId: str
    senderNick: str
    text: Union[TextMsg, None] = None
    content: Union[ContentMsg, None] = None
    msgtype: str
    sessionWebhookExpiredTime: int
    sessionWebhook: str

    conversationTitle: Union[str, None] = None
    senderPlatform: Union[str, None] = None
    atUsers: Union[list[dict], None] = None
    chatbotUserId: Union[str, None] = None
    isAdmin: Union[bool, None] = None
    conversationType: Union[str, None] = None
    createAt: Union[int, None] = None
    senderId: Union[str, None] = None
    isInAtList: Union[bool, None] = None
    robotCode: Union[str, None] = None


API_KEY = os.getenv("API_KEY")
QUERY_URL = os.getenv("QUERY_URL")
DINGTALK_TOKEN = os.getenv("DINGTALK_TOKEN")


def send_msg(
    url: str, session: str, content: str, title: str = None
) -> requests.Response:
    if title is None:
        return requests.post(
            url,
            json={
                "msgtype": "text",
                "text": {"content": content},
            },
            params={"session": session},
        )

    return requests.post(
        url,
        json={
            "msgtype": "markdown",
            "markdown": {
                "title": title,
                "text": content,
            },
        },
        params={"session": session},
    )


@app.post("/api/query")
async def query(
    request: QueryRequest, token: Annotated[Union[str, None], Header()] = None
):
    if token is None or token != DINGTALK_TOKEN:
        return JSONResponse(content={"msg": "Unauthorized"}, status_code=401)
    print(request)

    if request.msgtype == "text" and request.text is None:
        return JSONResponse(content={"msg": "Invalid request"}, status_code=400)
    elif request.msgtype in ["picture", "richText"] and (
        request.content is None
        or (request.content.richText is None and request.content.downloadCode is None)
    ):
        return JSONResponse(content={"msg": "Invalid request"}, status_code=400)

    query_content = ""
    if request.msgtype == "text":
        query_content = request.text.content
    elif request.msgtype == "richText":
        query_content = "\n".join(
            list(
                map(
                    lambda x: x.text,
                    filter(lambda x: x.text is not None, request.content.richText),
                )
            )
        )
    elif request.msgtype == "picture":
        pass

    webhook_splits = request.sessionWebhook.split("?")
    webhook_url, session = webhook_splits[0], webhook_splits[1].split("=")[1]

    def query_and_reply():
        resp = requests.post(
            f"{QUERY_URL}/v1/chat-messages",
            json={
                "query": query_content,
                "inputs": {},
                "response_mode": "blocking",
                "user": "user",
                "files": [],
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
        )
        if resp.status_code >= 200 and resp.status_code < 300:
            res_body = resp.json()
            # print(res_body)
            send_msg(
                webhook_url,
                session,
                f"@{request.senderNick}，你的提问是: {query_content}\n\n回答如下:\n\n{res_body['answer']}",
                title="xvx! 回答已生成!",
            )
        else:
            print(f"Failed to query: {resp.status_code} {resp.text}")
            send_msg(webhook_url, session, f"x_x 查询失败: {resp.status_code}")

    threading.Thread(target=query_and_reply, args=()).start()

    return {
        "text": {"content": f"x!x @{request.senderNick} 正在查询答案..."},
        "msgtype": "text",
    }
