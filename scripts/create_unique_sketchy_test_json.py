from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def norm_caption(text: str) -> str:
    return " ".join((text or "").strip().split())


def safe_stem(text: str, max_len: int = 80) -> str:
    text = norm_caption(text).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return (text or "segment")[:max_len].strip("_") or "segment"


def union_bbox(segments: list[dict[str, Any]]) -> dict[str, int] | None:
    boxes = [seg.get("bbox_coordinates") for seg in segments if seg.get("bbox_coordinates")]
    if not boxes:
        return None
    x1 = min(int(round(float(box["x1"]))) for box in boxes)
    y1 = min(int(round(float(box["y1"]))) for box in boxes)
    x2 = max(int(round(float(box["x2"]))) for box in boxes)
    y2 = max(int(round(float(box["y2"]))) for box in boxes)
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": max(0, x2 - x1),
        "height": max(0, y2 - y1),
    }


def merge_category(values: list[Any]) -> Any:
    unique = []
    for value in values:
        if value is not None and value not in unique:
            unique.append(value)
    if len(unique) == 1:
        return unique[0]
    if not unique:
        return None
    return unique


def load_mask(path: str) -> np.ndarray:
    mask = Image.open(path).convert("L")
    return np.array(mask, dtype=np.uint8) > 0


def save_union_mask(segments: list[dict[str, Any]], out_path: Path) -> None:
    masks = [load_mask(seg["instance_mask"]) for seg in segments]
    first_shape = masks[0].shape
    for seg, mask in zip(segments, masks):
        if mask.shape != first_shape:
            raise ValueError(
                f"Mask shape mismatch for caption={segments[0].get('caption')!r}: "
                f"{seg.get('instance_mask')} has {mask.shape}, expected {first_shape}"
            )
    union = np.logical_or.reduce(masks)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((union.astype(np.uint8) * 255), mode="L").save(out_path)


def build_merged_segment(
    segments: list[dict[str, Any]],
    out_mask_path: Path,
) -> dict[str, Any]:
    merged = deepcopy(segments[0])
    merged["instance_mask"] = str(out_mask_path)

    annotation_ids = [seg.get("annotation_id") for seg in segments if seg.get("annotation_id") is not None]
    if len(annotation_ids) == 1:
        merged["annotation_id"] = annotation_ids[0]
    elif len(annotation_ids) > 1:
        merged["annotation_id"] = annotation_ids[0]
        merged["annotation_ids"] = annotation_ids

    category_id = merge_category([seg.get("category_id") for seg in segments])
    category_name = merge_category([seg.get("category_name") for seg in segments])
    if category_id is not None:
        merged["category_id"] = category_id
    if category_name is not None:
        merged["category_name"] = category_name

    bbox = union_bbox(segments)
    if bbox is not None:
        merged["bbox_coordinates"] = bbox

    if len(segments) > 1:
        merged["merged_from"] = [
            {
                "annotation_id": seg.get("annotation_id"),
                "instance_mask": seg.get("instance_mask"),
                "bbox_coordinates": seg.get("bbox_coordinates"),
            }
            for seg in segments
        ]
    return merged


def process_item(item: dict[str, Any], mask_root: Path) -> tuple[dict[str, Any], int, int]:
    item_out = deepcopy(item)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for seg in item.get("segment", []) or []:
        caption = norm_caption(seg.get("caption", ""))
        if not caption or not seg.get("instance_mask"):
            continue
        if caption not in grouped:
            order.append(caption)
        grouped[caption].append(seg)

    image_id = item.get("image_id")
    image_key = str(image_id if image_id is not None else Path(item.get("file_name") or item.get("original_filename", "image")).stem)
    item_mask_root = mask_root / image_key

    merged_segments = []
    num_groups_merged = 0
    num_segments_removed = 0
    for idx, caption in enumerate(order):
        segments = grouped[caption]
        ann_ids = [str(seg.get("annotation_id")) for seg in segments if seg.get("annotation_id") is not None]
        suffix = "_".join(ann_ids) if ann_ids else f"{idx:04d}"
        out_name = f"{idx:04d}_{safe_stem(caption)}_{suffix}.png"
        out_mask_path = item_mask_root / out_name
        save_union_mask(segments, out_mask_path)
        merged_segments.append(build_merged_segment(segments, out_mask_path))

        if len(segments) > 1:
            num_groups_merged += 1
            num_segments_removed += len(segments) - 1

    item_out["segment"] = merged_segments
    return item_out, num_groups_merged, num_segments_removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-json",
        default="/mnt/data/zruan/kqy/pami/segmentation/SKETCHY_test_instance_mask.json",
    )
    parser.add_argument(
        "--output-json",
        default="/mnt/data/zruan/kqy/pami/segmentation/Sketchy_test_instance_new.json",
    )
    parser.add_argument(
        "--mask-root",
        default="/mnt/data/zruan/kqy/pami/test/unique_instance_masks",
    )
    args = parser.parse_args()

    input_json = Path(args.input_json)
    output_json = Path(args.output_json)
    mask_root = Path(args.mask_root)

    with input_json.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    out_records = []
    total_groups_merged = 0
    total_segments_removed = 0
    original_segments = 0
    new_segments = 0
    for item in records:
        original_segments += len(item.get("segment", []) or [])
        item_out, groups_merged, segments_removed = process_item(item, mask_root)
        total_groups_merged += groups_merged
        total_segments_removed += segments_removed
        new_segments += len(item_out.get("segment", []) or [])
        out_records.append(item_out)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(out_records, handle, ensure_ascii=False)

    print(json.dumps({
        "input_json": str(input_json),
        "output_json": str(output_json),
        "mask_root": str(mask_root),
        "images": len(out_records),
        "original_segments": original_segments,
        "new_segments": new_segments,
        "groups_merged": total_groups_merged,
        "segments_removed": total_segments_removed,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
