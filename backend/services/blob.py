import asyncio
import json
from typing import Any

import boto3
from botocore.exceptions import ClientError

from backend.config import settings


def create_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
    )


def ensure_bucket(client, bucket_name: str) -> None:
    try:
        client.head_bucket(Bucket=bucket_name)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchBucket"):
            client.create_bucket(Bucket=bucket_name)
        else:
            raise


def upload_json(client, key: str, data: Any, bucket: str | None = None) -> str:
    bucket = bucket or settings.s3_bucket_name
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, default=str).encode(),
        ContentType="application/json",
    )
    return key


def download_json(client, key: str, bucket: str | None = None) -> Any:
    bucket = bucket or settings.s3_bucket_name
    response = client.get_object(Bucket=bucket, Key=key)
    return json.loads(response["Body"].read().decode())


async def async_upload_json(key: str, data: Any, bucket: str | None = None) -> str:
    client = create_s3_client()
    return await asyncio.to_thread(upload_json, client, key, data, bucket)


async def async_download_json(key: str, bucket: str | None = None) -> Any:
    client = create_s3_client()
    return await asyncio.to_thread(download_json, client, key, bucket)
