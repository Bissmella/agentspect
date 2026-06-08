from celery import Celery

from backend.config import settings

celery = Celery("ata")

celery.conf.update(
    broker_url=settings.redis_url,
    result_backend=settings.redis_url,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

celery.autodiscover_tasks(["backend"])
