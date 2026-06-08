import json
from unittest.mock import MagicMock, patch

import pytest

from backend.pubsub import (
    HISTORY_TTL_SECONDS,
    _channel,
    _history_key,
    publish_event,
)


class TestChannelNaming:
    def test_channel_format(self):
        assert _channel("abc-123") == "ata:suite:abc-123"

    def test_history_key_format(self):
        assert _history_key("abc-123") == "ata:suite:abc-123:history"


class TestPublishEvent:
    @patch("backend.pubsub.redis.Redis")
    def test_publishes_and_stores_history(self, mock_redis_cls):
        mock_client = MagicMock()
        mock_redis_cls.from_url.return_value = mock_client

        from backend.pubsub import _seq_counters
        _seq_counters.pop("suite-1", None)

        publish_event("suite-1", "suite_started", {"key": "val"})

        mock_client.publish.assert_called_once()
        channel_arg = mock_client.publish.call_args[0][0]
        assert channel_arg == "ata:suite:suite-1"

        payload = json.loads(mock_client.publish.call_args[0][1])
        assert payload["event"] == "suite_started"
        assert payload["data"] == {"key": "val"}
        assert payload["seq"] == 1
        assert "timestamp" in payload

        mock_client.rpush.assert_called_once()
        mock_client.expire.assert_called_once_with(
            "ata:suite:suite-1:history", HISTORY_TTL_SECONDS
        )
        mock_client.close.assert_called_once()

    @patch("backend.pubsub.redis.Redis")
    def test_seq_increments(self, mock_redis_cls):
        mock_client = MagicMock()
        mock_redis_cls.from_url.return_value = mock_client

        from backend.pubsub import _seq_counters
        _seq_counters.pop("suite-2", None)

        publish_event("suite-2", "event_a")
        publish_event("suite-2", "event_b")

        calls = mock_client.publish.call_args_list
        payload_a = json.loads(calls[0][0][1])
        payload_b = json.loads(calls[1][0][1])
        assert payload_a["seq"] == 1
        assert payload_b["seq"] == 2

    @patch("backend.pubsub.redis.Redis")
    def test_default_empty_data(self, mock_redis_cls):
        mock_client = MagicMock()
        mock_redis_cls.from_url.return_value = mock_client

        from backend.pubsub import _seq_counters
        _seq_counters.pop("suite-3", None)

        publish_event("suite-3", "test_event")

        payload = json.loads(mock_client.publish.call_args[0][1])
        assert payload["data"] == {}
