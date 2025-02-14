"""QA API."""

import json
import os

from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

################ Assistant ################
LLM_URL = os.getenv("LLM_URL")
if not LLM_URL:
    raise RuntimeError("env `LLM_URL` is not specified !")

LLM_NAME = os.getenv("LLM_NAME")
if not LLM_NAME:
    raise RuntimeError("env `LLM_NAME` is not specified !")

LLM = ChatOpenAI(
    openai_api_key="EMPTY",
    openai_api_base=LLM_URL,
    model_name=LLM_NAME,
    temperature=0.2,
)

RESPONSE_SYNTHESIZING_PROMPT_TMPL = """\
# **Instructions**: 
1. You are good at French.
2. You are provided criminal documents which are written in French.
3. Your task is to answer the user question in French based on the provided documents.
4. Whenever possible, cite the relevant document to support your response.

## **Criminal documents**:
{document}
"""
RESPONSE_SYNTHESIZING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            RESPONSE_SYNTHESIZING_PROMPT_TMPL,
        ),
        ("user", "{question}"),
    ]
)

assistant = RESPONSE_SYNTHESIZING_PROMPT | LLM

################ API ################
app = FastAPI()


class QARequest(BaseModel):  # noqa: D101
    document: str
    question: str


@app.post("/v0/ask")
async def ask(request: QARequest):
    """Ask assistant."""
    if request.question and request.document:

        async def response_streamer():
            async for token in assistant.astream(request.dict()):
                yield json.dumps({"token": token.content}) + "\n"

        return StreamingResponse(response_streamer(), media_type="application/json")
    else:
        return Response(status_code=400)
