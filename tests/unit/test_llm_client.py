"""Unit tests for the LLM client abstraction."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from backend.llm.client import (
    AnthropicClient,
    GoogleClient,
    LLMResponse,
    OpenAIClient,
    create_llm_client,
)


class SampleOutput(BaseModel):
    answer: str
    confidence: float


class TestLLMResponse:
    def test_llm_response_basic(self):
        response = LLMResponse(content="Hello world")
        assert response.content == "Hello world"
        assert response.tool_calls is None
        assert response.usage is None

    def test_llm_response_with_tools(self):
        response = LLMResponse(
            content="",
            tool_calls=[{"name": "test_tool", "arguments": {"arg1": "value1"}, "id": "123"}],
            usage={"input_tokens": 10, "output_tokens": 20},
        )
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "test_tool"


class TestCreateLLMClient:
    def test_create_anthropic_client(self):
        with patch("anthropic.AsyncAnthropic"):
            client = create_llm_client("anthropic", "claude-sonnet-4-20250514")
            assert isinstance(client, AnthropicClient)
            assert client.model == "claude-sonnet-4-20250514"

    def test_create_openai_client(self):
        with patch("openai.AsyncOpenAI"):
            client = create_llm_client("openai", "gpt-4")
            assert isinstance(client, OpenAIClient)
            assert client.model == "gpt-4"

    def test_create_google_client(self):
        with patch("google.generativeai.configure"):
            client = create_llm_client("google", "gemini-pro")
            assert isinstance(client, GoogleClient)
            assert client.model_name == "gemini-pro"

    def test_create_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_client("unknown", "model")


class TestAnthropicClient:
    @pytest.fixture
    def mock_anthropic(self):
        with patch("anthropic.AsyncAnthropic") as mock:
            yield mock

    async def test_chat_basic(self, mock_anthropic):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Hello!")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic.return_value = mock_client

        client = AnthropicClient("claude-sonnet-4-20250514", api_key="test-key")
        response = await client.chat([{"role": "user", "content": "Hi"}])

        assert response.content == "Hello!"
        assert response.usage == {"input_tokens": 10, "output_tokens": 5}

    async def test_chat_with_system_message(self, mock_anthropic):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Response")]
        mock_response.usage.input_tokens = 15
        mock_response.usage.output_tokens = 8

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic.return_value = mock_client

        client = AnthropicClient("claude-sonnet-4-20250514", api_key="test-key")
        await client.chat([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ])

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "system" in call_kwargs
        assert call_kwargs["system"] == "You are helpful."

    async def test_chat_with_tool_call(self, mock_anthropic):
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "get_weather"
        mock_tool_block.input = {"city": "Paris"}
        mock_tool_block.id = "tool_123"

        mock_response = MagicMock()
        mock_response.content = [mock_tool_block]
        mock_response.usage.input_tokens = 20
        mock_response.usage.output_tokens = 10

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic.return_value = mock_client

        client = AnthropicClient("claude-sonnet-4-20250514", api_key="test-key")
        response = await client.chat(
            [{"role": "user", "content": "What's the weather in Paris?"}],
            tools=[{"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}],
        )

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "get_weather"
        assert response.tool_calls[0]["arguments"] == {"city": "Paris"}


class TestOpenAIClient:
    @pytest.fixture
    def mock_openai(self):
        with patch("openai.AsyncOpenAI") as mock:
            yield mock

    async def test_chat_basic(self, mock_openai):
        mock_message = MagicMock()
        mock_message.content = "Hello from OpenAI!"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 12
        mock_response.usage.completion_tokens = 6

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.return_value = mock_client

        client = OpenAIClient("gpt-4", api_key="test-key")
        response = await client.chat([{"role": "user", "content": "Hi"}])

        assert response.content == "Hello from OpenAI!"
        assert response.usage == {"input_tokens": 12, "output_tokens": 6}

    async def test_chat_with_tool_call(self, mock_openai):
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "search"
        mock_tool_call.function.arguments = '{"query": "test"}'
        mock_tool_call.id = "call_123"

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 15

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.return_value = mock_client

        client = OpenAIClient("gpt-4", api_key="test-key")
        response = await client.chat(
            [{"role": "user", "content": "Search for test"}],
            tools=[{"name": "search", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}],
        )

        assert response.tool_calls is not None
        assert response.tool_calls[0]["name"] == "search"
        assert response.tool_calls[0]["arguments"] == {"query": "test"}


class TestStructuredOutput:
    async def test_anthropic_structured_output(self):
        with patch("anthropic.AsyncAnthropic") as mock_anthropic:
            mock_tool_block = MagicMock()
            mock_tool_block.type = "tool_use"
            mock_tool_block.name = "structured_output"
            mock_tool_block.input = {"answer": "42", "confidence": 0.95}
            mock_tool_block.id = "tool_456"

            mock_response = MagicMock()
            mock_response.content = [mock_tool_block]
            mock_response.usage.input_tokens = 25
            mock_response.usage.output_tokens = 12

            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_anthropic.return_value = mock_client

            client = AnthropicClient("claude-sonnet-4-20250514", api_key="test-key")
            result = await client.chat_with_structured_output(
                [{"role": "user", "content": "What is the answer?"}],
                SampleOutput,
            )

            assert isinstance(result, SampleOutput)
            assert result.answer == "42"
            assert result.confidence == 0.95

    async def test_openai_structured_output(self):
        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_tool_call = MagicMock()
            mock_tool_call.function.name = "structured_output"
            mock_tool_call.function.arguments = '{"answer": "hello", "confidence": 0.8}'
            mock_tool_call.id = "call_789"

            mock_message = MagicMock()
            mock_message.content = None
            mock_message.tool_calls = [mock_tool_call]

            mock_choice = MagicMock()
            mock_choice.message = mock_message

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.usage.prompt_tokens = 30
            mock_response.usage.completion_tokens = 18

            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            client = OpenAIClient("gpt-4", api_key="test-key")
            result = await client.chat_with_structured_output(
                [{"role": "user", "content": "Say hello"}],
                SampleOutput,
            )

            assert isinstance(result, SampleOutput)
            assert result.answer == "hello"
            assert result.confidence == 0.8

    async def test_structured_output_no_tool_call_raises(self):
        with patch("anthropic.AsyncAnthropic") as mock_anthropic:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(type="text", text="No tool call")]
            mock_response.usage.input_tokens = 10
            mock_response.usage.output_tokens = 5

            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_anthropic.return_value = mock_client

            client = AnthropicClient("claude-sonnet-4-20250514", api_key="test-key")

            with pytest.raises(ValueError, match="did not return structured output"):
                await client.chat_with_structured_output(
                    [{"role": "user", "content": "Test"}],
                    SampleOutput,
                )
