from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
import json, os, re
from backend.llm.client import LLMClient, create_llm_client

app = FastAPI(title="LLM Mock Agent Under Test")

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

class ChatResponse(BaseModel):
    response: str
    session_id: str


def get_db(db_path: str) -> dict:
    if not os.path.exists(db_path):
        return {"users": [], "reservations": []}
    with open(db_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {"users": [], "reservations": []}

