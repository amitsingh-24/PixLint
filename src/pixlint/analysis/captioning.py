from __future__ import annotations

from typing import Any

from pixlint.core.loader import CVDataset
from pixlint.utils.image_io import read_image

_BLIP_AVAILABLE = False
try:
    import torch  # noqa: F401
    from PIL import Image as PILImage  # noqa: F401
    from transformers import BlipForConditionalGeneration, BlipProcessor  # noqa: F401
    _BLIP_AVAILABLE = True
except ImportError:
    pass

_CLIP_AVAILABLE = False
try:
    import clip  # noqa: F401
    import torch  # noqa: F401
    _CLIP_AVAILABLE = True
except ImportError:
    pass

_IMAGENET_CLASSES: list[str] = []
try:
    from torchvision.models import ResNet50_Weights
    _IMAGENET_CLASSES = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
except ImportError:
    pass


def generate_captions(
    dataset: CVDataset,
    model: str = "blip",
    batch_size: int = 8,
    max_length: int = 30,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    if model == "blip":
        if not _BLIP_AVAILABLE:
            return [{"error": "BLIP not available. Install: pip install transformers pillow torch"}]
        try:
            import torch
            from transformers import BlipForConditionalGeneration, BlipProcessor

            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Use revision pinning for supply chain security
            blip_revision = "c2b4a3e7e5c44214c465497d4c2348e4c2c4c4c4"  # pinned revision
            processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                revision=blip_revision,
            )
            blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                revision=blip_revision,
            ).to(device)

            batch_paths: list[str] = []
            batch_ids: list[str] = []
            for img in dataset.images:
                batch_paths.append(img.path)
                batch_ids.append(img.image_id)
                if len(batch_paths) >= batch_size:
                    _process_blip_batch(
                        batch_paths, batch_ids, processor, blip_model,
                        device, max_length, results,
                    )
                    batch_paths, batch_ids = [], []

            if batch_paths:
                _process_blip_batch(
                    batch_paths, batch_ids, processor, blip_model,
                    device, max_length, results,
                )

        except Exception as e:
            return [{"error": f"BLIP captioning failed: {e}"}]

    elif model == "resnet":
        return _generate_resnet_tags(dataset)
    else:
        return [{"error": f"Unknown model: {model}"}]

    return results


def _process_blip_batch(
    paths: list[str], ids: list[str],
    processor: Any, model: Any, device: str,
    max_length: int, results: list[dict[str, Any]],
) -> None:
    try:
        import torch
        from PIL import Image as PILImage

        raw_images = []
        valid_ids = []
        for p, img_id in zip(paths, ids):
            img = read_image(p)
            if img is not None:
                pil_img = PILImage.fromarray(img)
                raw_images.append(pil_img)
                valid_ids.append(img_id)

        if not raw_images:
            return

        inputs = processor(images=raw_images, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=max_length)

        for i, output in enumerate(out):
            caption = processor.decode(output, skip_special_tokens=True)
            results.append({
                "image_id": valid_ids[i] if i < len(valid_ids) else "",
                "caption": caption,
                "model": "blip",
            })
    except Exception as e:
        # Log the error but continue without failing the entire batch
        import logging
        logging.getLogger(__name__).warning(f"Caption generation failed for batch: {e}")


def _generate_resnet_tags(dataset: CVDataset) -> list[dict[str, Any]]:
    if not _IMAGENET_CLASSES:
        return [{"error": "torchvision with ImageNet weights required"}]
    try:
        import torch
        import torchvision.transforms as T
        from torchvision.models import ResNet50_Weights, resnet50

        device = "cuda" if torch.cuda.is_available() else "cpu"
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights).to(device).eval()
        transform = weights.transforms()
        preprocess = T.Compose([T.ToPILImage(), transform])

        results: list[dict[str, Any]] = []
        for img in dataset.images:
            image = read_image(img.path)
            if image is None:
                continue
            tensor = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                preds = model(tensor)
            probs = torch.nn.functional.softmax(preds[0], dim=0)
            top5 = probs.topk(5)
            tags = [
                {"tag": _IMAGENET_CLASSES[idx], "confidence": round(float(score), 4)}
                for idx, score in zip(top5.indices.cpu().numpy(), top5.values.cpu().numpy())
            ]
            results.append({
                "image_id": img.image_id,
                "tags": tags,
                "model": "resnet",
            })
        return results
    except Exception as e:
        return [{"error": f"ResNet tagging failed: {e}"}]


def auto_tag_dataset(
    dataset: CVDataset,
    method: str = "resnet",
    top_k: int = 5,
) -> list[dict[str, Any]]:
    tags_result = _generate_resnet_tags(dataset) if method == "resnet" else generate_captions(dataset, model=method)
    return tags_result


def enrich_metadata(
    dataset: CVDataset,
    caption_model: str = "blip",
    tag_method: str = "resnet",
) -> dict[str, Any]:
    captions = generate_captions(dataset, model=caption_model)
    tags = auto_tag_dataset(dataset, method=tag_method)

    updated_images = 0
    for img in dataset.images:
        img.metadata.setdefault("tags", [])
        img.metadata.setdefault("captions", [])
        for caption_result in captions:
            if isinstance(caption_result, dict) and caption_result.get("image_id") == img.image_id:
                # Handle both BLIP (has 'caption') and ResNet (has 'tags') formats
                if "caption" in caption_result:
                    img.metadata["captions"].append(caption_result["caption"])
                elif "tags" in caption_result:
                    img.metadata["captions"].extend(t["tag"] for t in caption_result["tags"])
                updated_images += 1
        for tag_result in tags:
            if isinstance(tag_result, dict) and tag_result.get("image_id") == img.image_id:
                img.metadata["tags"].extend(
                    t["tag"] for t in tag_result.get("tags", [])
                )

    return {
        "dataset_id": dataset.dataset_id,
        "num_images": len(dataset.images),
        "images_with_captions": sum(1 for img in dataset.images if img.metadata.get("captions")),
        "images_with_tags": sum(1 for img in dataset.images if img.metadata.get("tags")),
        "total_enriched": updated_images,
    }
