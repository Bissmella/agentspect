from backend.celery_app import celery


class TestCeleryConfig:
    def test_app_name(self):
        assert celery.main == "ata"

    def test_broker_configured(self):
        assert "redis" in celery.conf.broker_url

    def test_serializer_is_json(self):
        assert celery.conf.task_serializer == "json"
        assert celery.conf.result_serializer == "json"

    def test_track_started(self):
        assert celery.conf.task_track_started is True

    def test_acks_late(self):
        assert celery.conf.task_acks_late is True
