from __future__ import annotations

from unittest.mock import patch

import pytest

from pixlint.core.cloud import (
    _list_objects,
    _list_gcs_objects,
    _list_azure_objects,
    list_s3_objects,
)


class TestListS3Objects:
    def test_list_s3_no_boto3(self):
        with patch("pixlint.core.cloud._S3_AVAILABLE", False):
            result = list_s3_objects("test-bucket")
            assert result == []


class TestCloudFunctions:
    def test_list_objects_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            _list_objects("unknown", "b", "", False, None, None)

    def test_list_objects_s3_no_boto3(self):
        with patch("pixlint.core.cloud._S3_AVAILABLE", False):
            with pytest.raises(ImportError, match="boto3"):
                _list_objects("s3", "b", "", False, None, None)

    def test_list_gcs_no_gcs(self):
        with patch("pixlint.core.cloud._GCS_AVAILABLE", False):
            with pytest.raises(ImportError, match="google-cloud-storage"):
                _list_gcs_objects("bucket", "prefix", False)

    def test_list_azure_no_azure(self):
        with patch("pixlint.core.cloud._AZURE_AVAILABLE", False):
            with pytest.raises(ImportError, match="azure-storage-blob"):
                _list_azure_objects("bucket", "prefix", None)


class TestLoadCloud:
    def test_load_unknown_provider(self):
        from pixlint.core.cloud import load_cloud_dataset
        with pytest.raises(ValueError, match="Unknown provider"):
            load_cloud_dataset(provider="unknown", bucket="test")

    @patch("pixlint.core.cloud._S3_AVAILABLE", False)
    def test_load_s3_no_boto3(self):
        from pixlint.core.cloud import load_cloud_dataset
        with pytest.raises(ImportError, match="boto3"):
            load_cloud_dataset(provider="s3", bucket="test")

    @patch("pixlint.core.cloud._list_objects")
    def test_load_cloud_empty(self, mock_list):
        from pixlint.core.cloud import load_cloud_dataset
        mock_list.return_value = []
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            os.makedirs(os.path.join(tmpdir, "images"), exist_ok=True)
            with patch("pixlint.core.cloud.tempfile.mkdtemp", return_value=tmpdir):
                result = load_cloud_dataset(
                    provider="s3", bucket="test-bucket",
                )
                assert result is not None
                assert len(result.images) == 0
