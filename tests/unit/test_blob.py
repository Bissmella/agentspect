import pytest
from unittest.mock import MagicMock, patch

from backend.services.blob import (
    create_s3_client,
    download_json,
    ensure_bucket,
    upload_json,
)


@pytest.fixture
def mock_s3():
    client = MagicMock()
    return client


class TestEnsureBucket:
    def test_bucket_exists(self, mock_s3):
        mock_s3.head_bucket.return_value = {}
        ensure_bucket(mock_s3, "test-bucket")
        mock_s3.head_bucket.assert_called_once_with(Bucket="test-bucket")
        mock_s3.create_bucket.assert_not_called()

    def test_bucket_not_found_creates(self, mock_s3):
        from botocore.exceptions import ClientError

        mock_s3.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "404"}}, "HeadBucket"
        )
        ensure_bucket(mock_s3, "test-bucket")
        mock_s3.create_bucket.assert_called_once_with(Bucket="test-bucket")

    def test_other_error_raises(self, mock_s3):
        from botocore.exceptions import ClientError

        mock_s3.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "403"}}, "HeadBucket"
        )
        with pytest.raises(ClientError):
            ensure_bucket(mock_s3, "test-bucket")


class TestUploadJson:
    def test_upload_returns_key(self, mock_s3):
        key = upload_json(mock_s3, "test/key.json", {"hello": "world"}, bucket="b")
        assert key == "test/key.json"
        mock_s3.put_object.assert_called_once()
        call_kwargs = mock_s3.put_object.call_args[1]
        assert call_kwargs["Bucket"] == "b"
        assert call_kwargs["Key"] == "test/key.json"
        assert call_kwargs["ContentType"] == "application/json"

    def test_upload_serializes_data(self, mock_s3):
        upload_json(mock_s3, "k.json", {"nums": [1, 2, 3]}, bucket="b")
        body = mock_s3.put_object.call_args[1]["Body"]
        import json
        assert json.loads(body) == {"nums": [1, 2, 3]}


class TestDownloadJson:
    def test_download_returns_parsed(self, mock_s3):
        import json
        body_mock = MagicMock()
        body_mock.read.return_value = json.dumps({"key": "value"}).encode()
        mock_s3.get_object.return_value = {"Body": body_mock}

        result = download_json(mock_s3, "test/key.json", bucket="b")
        assert result == {"key": "value"}
        mock_s3.get_object.assert_called_once_with(Bucket="b", Key="test/key.json")


class TestCreateS3Client:
    @patch("backend.services.blob.boto3")
    def test_creates_client_from_settings(self, mock_boto3):
        create_s3_client()
        mock_boto3.client.assert_called_once_with(
            "s3",
            endpoint_url="http://localhost:9000",
            aws_access_key_id="minioadmin",
            aws_secret_access_key="minioadmin",
        )
