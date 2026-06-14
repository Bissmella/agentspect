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

def save_db(db_path:str, data: dict):
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)



def execute_tool(name: str, args: dict, db_path: str, disability: str |None = None) -> dict:
    db = get_db(db_path)

    if name == "get_user":
        phone = args.get("phone")
        for u in db.get("users", []):
            if u.get("phone") == phone:
                return {"status": "success", "user": u}
        return {"status": "not_found", "message": "User not found"}
    
    elif name == "create_user":
        if disability == "no_user_registration":
            return {"status": "error", "message": "User registration failed."}
        
        name_val = args.get("name")
        phone = args.get("phone")

        #check if already registered
        for u in db.get("users", []):
            if u.get("phone") == phone:
                return {"status": "error", "message": "user phone number already exists"}
        
        new_user = {"name": name_val, "phone": phone}
        db.setdefault("users", []).append(new_user)
        if disability != "no_save":
            save_db(db_path, db)
        return {"status": "success", "user": new_user}
    
    elif name == "create_reservation":
        phone = args.get("phone")
        slot = args.get("slot")

        if disability != "double_booking":
            for r in db.get("reservations", []):
                if r.get("slot") == slot:
                    return {"status": "error", "message": "slot is already booked"}
        
        new_res = {"phone": phone, "slot": slot}
        db.setdefault("reservations", []).append(new_res)
        if disability != "no_save":
            save_db(db_path, db)
        return {"status": "success", "reservation": new_res}

    return {"status": "error", "message": f"unknown tool: {name}"}




@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request, response: Response):
    db_path = getattr(req.app.state, "db_path", "llm_agent_db.json")
    disability = getattr(req.app.state, "disability", None)

    llm_client = getattr(req.app.state, "llm_client", None) or create_llm_client("openrouter", "openai/gpt-4o-mini")
    session_id = request.session_id or "default_session"

    if not hasattr(req.app.state, "sessions"):
        req.app.state.sessions = {}
    if session_id not in req.app.state.sessions:
        req.app.state.sessions[session_id] = []

    session_history = req.app.state.sessions[session_id]
    session_history.append({"role": "user", "content": request.message})

    system_prompt = """
    ou are a reservation assistant. Available slots: "2026-05-20T10:00" and "2026-05-20T14:00".
    Rules:
    1. Always ask for user's phone number first.
    2. Check if user is registered using get_user.
    3. If not registered, ask for name and run create_user.
    4. Book reservation using create_reservation once you have user's phone number and the slot.
    
    """

    messages = [{"role": "system", "content": system_prompt}] + session_history

    tools = [
        {"name": "get_user", "description": "Lookup user", "parameters": {"type": "object", "properties": {"phone": {"type": "string"}}, "required": ["phone"]}},
        {"name": "create_user", "description": "Add user", "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "phone": {"type": "string"}}, "required": ["name", "phone"]}},
        {"name": "create_reservation", "description": "Add booking", "parameters": {"type": "object", "properties": {"phone": {"type": "string"}, "slot": {"type": "string"}}, "required": ["phone", "slot"]}}
    ]

    for _ in range(5):
        res_llm = await llm_client.chat(messages, tools=tools, temperature=0.0)
        if not res_llm.tool_calls:
            session_history.append({"role": "assistant", "content": res_llm.content})
            return ChatResponse(response=res_llm.content, session_id=session_id)

        openai_tool_calls = [
            {
                "id": tc.get("id", tc["name"]),
                "type": "function",
                "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}
            }
            for tc in res_llm.tool_calls
        ]
        messages.append({"role": "assistant", "content": res_llm.content or "", "tool_calls": openai_tool_calls})

        for tool_call in res_llm.tool_calls:
            name = tool_call["name"]
            args = tool_call["arguments"]
            res_id = tool_call.get("id", name)
            tool_result = execute_tool(name, args, db_path, disability)
            messages.append({"role": "tool", "content": json.dumps(tool_result), "tool_call_id": res_id})

    fallback = "I was unable to complete your request after multiple attempts."
    session_history.append({"role": "assistant", "content": fallback})
    return ChatResponse(response=fallback, session_id=session_id)