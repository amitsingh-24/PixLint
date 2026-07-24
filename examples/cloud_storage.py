#!/usr/bin/env python3
"""
PixLint — Cloud Storage (S3/GCS/Azure)

Load datasets directly from cloud buckets. Credentials from env vars only.
"""

from pixlint.core.cloud import load_cloud_dataset

# REQUIRED: Credentials from environment (never from params!)
# AWS:
# export AWS_ACCESS_KEY_ID=...
# export AWS_SECRET_ACCESS_KEY=...
# export AWS_DEFAULT_REGION=us-east-1
#
# GCS:
# export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json
#
# Azure:
# export AZURE_STORAGE_CONNECTION_STRING=...

def main():
    # S3 Example
    print("Loading from S3...")
    ds = load_cloud_dataset(
        provider="s3",
        bucket="my-cv-datasets",
        prefix="coco/train/",
        cache_dir="/tmp/s3_cache",  # Local cache
        name="s3_coco_train"
    )
    print(f"  Loaded: {ds.name} ({len(ds)} images)")

    # GCS Example
    # ds = load_cloud_dataset(
    #     provider="gcs",
    #     bucket="my-bucket",
    #     prefix="voc/",
    #     name="gcs_voc"
    # )

    # Azure Example
    # ds = load_cloud_dataset(
    #     provider="azure",
    #     bucket="my-container",
    #     prefix="yolo/",
    #     connection_string=os.environ["AZURE_STORAGE_CONNECTION_STRING"],
    #     name="azure_yolo"
    # )

if __name__ == "__main__":
    main()
