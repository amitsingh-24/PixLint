from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from pixlint.core.loader import CVDataset, load_dataset


_S3_AVAILABLE = False
try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    _S3_AVAILABLE = True
except ImportError:
    pass

_GCS_AVAILABLE = False
try:
    from google.cloud import storage
    _GCS_AVAILABLE = True
except ImportError:
    pass

_AZURE_AVAILABLE = False
try:
    from azure.storage.blob import BlobServiceClient
    _AZURE_AVAILABLE = True
except ImportError:
    pass


def _init_s3_client(anon: bool = False, endpoint_url: str | None = None) -> Any:
    config = Config(signature_version=UNSIGNED) if anon else None
    return boto3.client("s3", config=config, endpoint_url=endpoint_url)


def list_s3_objects(
    bucket: str,
    prefix: str = "",
    anon: bool = False,
    endpoint_url: str | None = None,
) -> list[dict[str, Any]]:
    if not _S3_AVAILABLE:
        return []
    client = _init_s3_client(anon=anon, endpoint_url=endpoint_url)
    paginator = client.get_paginator("list_objects_v2")
    results: list[dict[str, Any]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("/"):
                continue
            results.append({"key": obj["Key"], "size": obj["Size"], "etag": obj.get("ETag", "")})
    return results


def load_cloud_dataset(
    provider: str = "s3",
    bucket: str = "",
    prefix: str = "",
    anon: bool = False,
    endpoint_url: str | None = None,
    connection_string: str | None = None,
    cache_dir: str | None = None,
    name: str | None = None,
) -> CVDataset:
    cache_path = Path(cache_dir or tempfile.mkdtemp(prefix=f"{provider}_cache_"))
    cache_path.mkdir(parents=True, exist_ok=True)
    images_dir = cache_path / "images"
    images_dir.mkdir(exist_ok=True)
    lock = threading.Lock()
    failed: list[str] = []

    objects = _list_objects(provider, bucket, prefix, anon, endpoint_url, connection_string)

    def _download(obj: dict[str, Any]) -> str | None:
        key = obj["key"]
        stem = key.replace("/", "_").replace("\\", "_")
        local_path = str(images_dir / stem)
        if os.path.exists(local_path):
            return local_path
        with lock:
            if os.path.exists(local_path):
                return local_path
            try:
                if provider == "s3":
                    client = _init_s3_client(anon=anon, endpoint_url=endpoint_url)
                    client.download_file(bucket, key, local_path)
                elif provider == "gcs":
                    _download_gcs(bucket, key, local_path, anon)
                elif provider == "azure":
                    _download_azure(bucket, prefix, key, local_path, connection_string)
                return local_path
            except Exception:
                failed.append(key)
                return None

    for obj in objects:
        _download(obj)

    return load_dataset(str(images_dir), format="folder", name=name or f"{provider}_{bucket}_{prefix}")


def _list_objects(
    provider: str, bucket: str, prefix: str, anon: bool,
    endpoint_url: str | None, connection_string: str | None,
) -> list[dict[str, Any]]:
    if provider == "s3":
        if not _S3_AVAILABLE:
            raise ImportError("boto3 required. pip install boto3")
        return list_s3_objects(bucket, prefix, anon=anon, endpoint_url=endpoint_url)
    elif provider == "gcs":
        return _list_gcs_objects(bucket, prefix, anon)
    elif provider == "azure":
        return _list_azure_objects(bucket, prefix, connection_string)
    raise ValueError(f"Unknown provider: {provider}")


def _list_gcs_objects(bucket: str, prefix: str, anon: bool) -> list[dict[str, Any]]:
    if not _GCS_AVAILABLE:
        raise ImportError("google-cloud-storage is required for GCS support")
    client = storage.Client.create_anonymous_client() if anon else storage.Client()
    bucket_obj = client.bucket(bucket)
    blobs = bucket_obj.list_blobs(prefix=prefix)
    return [{"key": b.name, "size": b.size, "etag": b.etag} for b in blobs if not b.name.endswith("/")]


def _list_azure_objects(bucket: str, prefix: str, connection_string: str | None) -> list[dict[str, Any]]:
    if not _AZURE_AVAILABLE:
        raise ImportError("azure-storage-blob is required for Azure support")
    if connection_string:
        service = BlobServiceClient.from_connection_string(connection_string)
    else:
        service = BlobServiceClient(account_url=f"https://{bucket}.blob.core.windows.net")
    container_client = service.get_container_client(prefix or "$default")
    blobs = container_client.list_blobs()
    return [{"key": b.name, "size": b.size} for b in blobs if not b.name.endswith("/")]


def _download_gcs(bucket: str, key: str, local_path: str, anon: bool) -> None:
    if not _GCS_AVAILABLE:
        raise ImportError("google-cloud-storage is required for GCS support")
    client = storage.Client.create_anonymous_client() if anon else storage.Client()
    bucket_obj = client.bucket(bucket)
    bucket_obj.blob(key).download_to_filename(local_path)


def _download_azure(
    bucket: str, prefix: str, key: str, local_path: str, connection_string: str | None,
) -> None:
    if not _AZURE_AVAILABLE:
        raise ImportError("azure-storage-blob is required for Azure support")
    if connection_string:
        service = BlobServiceClient.from_connection_string(connection_string)
    else:
        service = BlobServiceClient(account_url=f"https://{bucket}.blob.core.windows.net")
    container_client = service.get_container_client(prefix or "$default")
    container_client.get_blob_client(key).download_blob().readinto(open(local_path, "wb"))
