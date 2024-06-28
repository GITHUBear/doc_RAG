from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from search_engine import get_model
from config import RagConfig
from chat import chat, multi_chat
import logging
import json
import datetime

import os
from zhipuai import ZhipuAI
from dotenv import load_dotenv
from config import RagConfig

load_dotenv(".env")

handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(
    logging.DEBUG if os.environ.get("DEBUG", "false") == "true" else logging.INFO
)

app = FastAPI()

# model = get_model(RagConfig())


@app.get("/")
async def root():
    return {"message": "Hello World"}


class EmbeddingRequest(BaseModel):
    input: Union[list[str], str, None] = None
    model: str = "BAAI/bge-m3"
    encoding_format: str = "float"


class RerankRequest(BaseModel):
    model: str = "BAAI/bge-m3"
    query: str
    documents: list[str]
    top_n: Union[int, None] = None


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/embeddings")
async def embedding(request: EmbeddingRequest):
    print(request.input)
    if request.input is None or type(request.input) == str or len(request.input) == 0:
        return {
            "object": "list",
            "data": [],
            "model": "BAAI/bge-m3",
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0,
            },
        }
    resp = [
        {
            "object": "embedding",
            "index": 0,
            "embedding": model.encode(st, batch_size=6, max_length=8192)[
                "dense_vecs"
            ].tolist(),
        }
        for st in request.input
    ]
    return {
        "object": "list",
        "data": resp,
        "model": "BAAI/bge-m3",
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.post("/legacy/embeddings")  # Legacy endpoint for a mistake in the frontend
async def legacy_embedding(request: EmbeddingRequest):
    print(request.input)
    if request.input is None or type(request.input) == str or len(request.input) == 0:
        return {
            "object": "list",
            "data": [],
            "model": "GAAI/bge-m3",
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0,
            },
        }
    resp = [
        {
            "object": "embedding",
            "index": 0,
            "embedding": model.encode(st, batch_size=6, max_length=8192)[
                "dense_vecs"
            ].tolist(),
        }
        for st in request.input
    ]
    return {
        "object": "list",
        "data": resp,
        "model": "GAAI/bge-m3",
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.post("/rerank")
async def rerank(request: RerankRequest):
    if request.documents is None or len(request.documents) == 0:
        return {
            "model": "BAAI/bge-m3",
            "query": request.query,
            "results": [],
        }
    scores = [
        model.compute_score(
            [request.query, st],
            batch_size=8,
            weights_for_different_modes=[0.4, 0.2, 0.4],
        )["colbert+sparse+dense"]
        for st in request.documents
    ]

    results = [
        {
            "index": i,
            "document": {
                "text": st,
            },
            "relevance_score": scores[i],
        }
        for i, st in enumerate(request.documents)
    ]
    return {
        "model": "BAAI/bge-m3",
        "query": request.query,
        "results": results,
    }


@app.post("/chat")
async def handle_chat(request: Request):
    data = await request.json()
    response = chat(data["input"])
    return {"answer": response}


@app.post("/api/chat")
async def handle_api_chat(request: Request):
    data = await request.json()
    logger.debug(data)

    messages: list = data["messages"]
    logger.debug(messages)

    query = messages.pop()["content"]

    generator = multi_chat(query, messages, stream=True, model="tongyi")
    return StreamingResponse(generator, media_type="application/json")
