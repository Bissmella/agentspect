from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from backend.config import settings


class LLMResponse(BaseModel):
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    usage: dict[str, int] | None = None


class LLMClient(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        pass

    @abstractmethod
    async def chat_with_structured_output(
        self,
        messages: list[dict[str, Any]],
        output_schema: type[BaseModel],
        temperature: float = 0.0,
    ) -> BaseModel:
        pass


class AnthropicClient(LLMClient):
    def __init__(self, model: str, api_key: str | None = None):
        import anthropic

        self.model = model
        self.client = anthropic.AsyncAnthropic(api_key=api_key or settings.anthropic_api_key)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        system_message = None
        chat_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                chat_messages.append(msg)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system_message:
            kwargs["system"] = system_message

        if tools:
            kwargs["tools"] = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t["parameters"],
                }
                for t in tools
            ]

        response = await self.client.messages.create(**kwargs)

        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                tool_calls.append({"name": block.name, "arguments": block.input, "id": block.id})

        return LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            usage={"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
        )

    async def chat_with_structured_output(
        self,
        messages: list[dict[str, Any]],
        output_schema: type[BaseModel],
        temperature: float = 0.0,
    ) -> BaseModel:
        schema = output_schema.model_json_schema()
        tool = {
            "name": "structured_output",
            "description": f"Return output conforming to {output_schema.__name__}",
            "parameters": schema,
        }

        response = await self.chat(messages, tools=[tool], temperature=temperature)

        if response.tool_calls:
            return output_schema.model_validate(response.tool_calls[0]["arguments"])

        raise ValueError("LLM did not return structured output via tool call")


class OpenAIClient(LLMClient):
    def __init__(self, model: str, api_key: str | None = None):
        import openai

        self.model = model
        self.client = openai.AsyncOpenAI(api_key=api_key or settings.openai_api_key)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if tools:
            kwargs["tools"] = [
                {"type": "function", "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t["parameters"]}}
                for t in tools
            ]

        response = await self.client.chat.completions.create(**kwargs)

        message = response.choices[0].message
        content = message.content or ""
        tool_calls = None

        if message.tool_calls:
            import json

            tool_calls = [
                {"name": tc.function.name, "arguments": json.loads(tc.function.arguments), "id": tc.id}
                for tc in message.tool_calls
            ]

        usage = None
        if response.usage:
            usage = {"input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens}

        return LLMResponse(content=content, tool_calls=tool_calls, usage=usage)

    async def chat_with_structured_output(
        self,
        messages: list[dict[str, Any]],
        output_schema: type[BaseModel],
        temperature: float = 0.0,
    ) -> BaseModel:
        schema = output_schema.model_json_schema()
        tool = {
            "name": "structured_output",
            "description": f"Return output conforming to {output_schema.__name__}",
            "parameters": schema,
        }

        response = await self.chat(messages, tools=[tool], temperature=temperature)

        if response.tool_calls:
            return output_schema.model_validate(response.tool_calls[0]["arguments"])

        raise ValueError("LLM did not return structured output via tool call")


class GoogleClient(LLMClient):
    def __init__(self, model: str, api_key: str | None = None):
        import google.generativeai as genai

        self.model_name = model
        genai.configure(api_key=api_key or settings.google_api_key)
        self.genai = genai

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        import asyncio

        model = self.genai.GenerativeModel(self.model_name)

        contents = []
        for msg in messages:
            role = "user" if msg["role"] in ("user", "system") else "model"
            contents.append({"role": role, "parts": [msg["content"]]})

        generation_config = self.genai.GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)

        tool_declarations = None
        if tools:
            tool_declarations = [
                self.genai.protos.Tool(
                    function_declarations=[
                        self.genai.protos.FunctionDeclaration(
                            name=t["name"],
                            description=t.get("description", ""),
                            parameters=self._schema_to_proto(t["parameters"]),
                        )
                        for t in tools
                    ]
                )
            ]

        response = await asyncio.to_thread(
            model.generate_content,
            contents,
            generation_config=generation_config,
            tools=tool_declarations,
        )

        content = ""
        tool_calls = []

        for part in response.parts:
            if part.text:
                content = part.text
            if part.function_call:
                tool_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                    "id": part.function_call.name,
                })

        return LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            usage=None,
        )

    def _schema_to_proto(self, schema: dict[str, Any]) -> Any:
        type_mapping = {
            "string": self.genai.protos.Type.STRING,
            "integer": self.genai.protos.Type.INTEGER,
            "number": self.genai.protos.Type.NUMBER,
            "boolean": self.genai.protos.Type.BOOLEAN,
            "array": self.genai.protos.Type.ARRAY,
            "object": self.genai.protos.Type.OBJECT,
        }

        json_type = schema.get("type", "object")
        proto_type = type_mapping.get(json_type, self.genai.protos.Type.OBJECT)

        properties = {}
        if "properties" in schema:
            for name, prop in schema["properties"].items():
                properties[name] = self._schema_to_proto(prop)

        items = None
        if json_type == "array" and "items" in schema:
            items = self._schema_to_proto(schema["items"])

        return self.genai.protos.Schema(
            type=proto_type,
            properties=properties if properties else None,
            items=items,
            required=schema.get("required", []),
            description=schema.get("description", ""),
        )

    async def chat_with_structured_output(
        self,
        messages: list[dict[str, Any]],
        output_schema: type[BaseModel],
        temperature: float = 0.0,
    ) -> BaseModel:
        schema = output_schema.model_json_schema()
        tool = {
            "name": "structured_output",
            "description": f"Return output conforming to {output_schema.__name__}",
            "parameters": schema,
        }

        response = await self.chat(messages, tools=[tool], temperature=temperature)

        if response.tool_calls:
            return output_schema.model_validate(response.tool_calls[0]["arguments"])

        raise ValueError("LLM did not return structured output via tool call")


def create_llm_client(provider: str, model: str, api_key: str | None = None) -> LLMClient:
    providers = {
        "anthropic": AnthropicClient,
        "openai": OpenAIClient,
        "google": GoogleClient,
    }

    if provider not in providers:
        raise ValueError(f"Unknown LLM provider: {provider}. Must be one of: {list(providers.keys())}")

    return providers[provider](model=model, api_key=api_key)
