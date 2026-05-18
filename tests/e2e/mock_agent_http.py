"""
Mock HTTP agent for testing the HTTP protocol adapter.

A trivial FastAPI app that follows a scripted conversation.
Can be used as a test fixture or run standalone for manual testing.
"""

from fastapi import FastAPI, Request, Response
from pydantic import BaseModel

app = FastAPI(title="Mock HTTP Agent")


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


_sessions: dict[str, list[str]] = {}

_scripts: dict[str, list[str]] = {
    "default": [
        "Hello! How can I help you today?",
        "I understand. Let me look into that for you.",
        "I've found the information you requested.",
        "Is there anything else I can help you with?",
        "Goodbye! Have a great day.",
    ],
    "booking": [
        "Welcome to the booking assistant. What date would you like to book?",
        "Let me check availability for that date.",
        "That slot is available. Would you like me to book it?",
        "Your booking has been confirmed. Confirmation number: BK12345.",
        "Thank you for using our service!",
    ],
    "refusal": [
        "I'm sorry, I cannot help with that request.",
        "That action is not permitted.",
        "I must decline. Is there something else I can help with?",
    ],
    "error": [
        "ERROR_RESPONSE",
    ],
}


def get_script(session_id: str) -> str:
    if "booking" in session_id.lower():
        return "booking"
    elif "refuse" in session_id.lower():
        return "refusal"
    elif "error" in session_id.lower():
        return "error"
    return "default"


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request, response: Response):
    session_id = request.session_id or req.cookies.get("session_id", "default-session")

    if session_id not in _sessions:
        _sessions[session_id] = []

    _sessions[session_id].append(request.message)
    turn_index = len(_sessions[session_id]) - 1

    script_name = get_script(session_id)
    script = _scripts[script_name]

    if turn_index < len(script):
        agent_response = script[turn_index]
    else:
        agent_response = "I don't have anything more to say."

    response.set_cookie("session_id", session_id)

    return ChatResponse(response=agent_response, session_id=session_id)


@app.post("/reset")
async def reset():
    _sessions.clear()
    return {"status": "reset"}


@app.get("/sessions")
async def get_sessions():
    return {"sessions": _sessions}


def set_script(name: str, responses: list[str]) -> None:
    _scripts[name] = responses


def clear_sessions() -> None:
    _sessions.clear()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765)
