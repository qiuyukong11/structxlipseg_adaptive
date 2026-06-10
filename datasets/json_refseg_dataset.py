from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


CLIP_MEAN = (0.485, 0.456, 0.406)
CLIP_STD = (0.229, 0.224, 0.225)
STRUCTURE_FALLBACK_FIELDS = (
    "filename_diffusion_cropped",
    "filename_diffusion_sketch",
    "filename_sketch",
    "filename_canny_sketch",
    "filename_sketch_canny",
    "filename_diffusion",
    "filename_canny",
    "filename_hed",
    "filename_log",
)
ORIGINAL_STRUCTURE_FALLBACK_FIELDS = (
    "original_filename_diffusion",
    "original_filename_diffusion_sketch",
    "original_filename_sketch",
    "original_filename_canny_sketch",
    "original_filename_canny",
)


def _resolve(path: str, data_root: str = "") -> str:
    p = Path(path or "")
    if not path:
        return ""
    if p.is_absolute() or not data_root:
        return str(p)
    return str(Path(data_root) / p)


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _bbox_to_ints(bbox: Optional[Dict]) -> Optional[tuple[int, int, int, int]]:
    if not bbox:
        return None
    try:
        x1 = int(round(float(bbox["x1"])))
        y1 = int(round(float(bbox["y1"])))
        x2 = int(round(float(bbox["x2"])))
        y2 = int(round(float(bbox["y2"])))
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _bbox_size(bbox: Optional[Dict]) -> Optional[tuple[int, int]]:
    xyxy = _bbox_to_ints(bbox)
    if xyxy is None:
        return None
    x1, y1, x2, y2 = xyxy
    return x2 - x1, y2 - y1


def _size_matches_bbox(mask_size: tuple[int, int], bbox: Optional[Dict], tolerance: int = 2) -> bool:
    bbox_size = _bbox_size(bbox)
    if bbox_size is None:
        return False
    return abs(mask_size[0] - bbox_size[0]) <= tolerance and abs(mask_size[1] - bbox_size[1]) <= tolerance


def _unpad_square_canvas_mask(raw: Image.Image, image_size_wh: tuple[int, int]) -> Image.Image:
    width, height = image_size_wh
    side = raw.size[0]
    scale = min(side / width, side / height)
    resized_w = int(round(width * scale))
    resized_h = int(round(height * scale))
    left = 0
    top = 0
    if resized_w < side:
        diff = side - resized_w
        left = diff // 2 + (diff % 2)
    if resized_h < side:
        diff = side - resized_h
        top = diff // 2 + (diff % 2)
    crop = raw.crop((left, top, left + resized_w, top + resized_h))
    crop = crop.resize((width, height), resample=Image.Resampling.NEAREST)
    return crop.point(lambda p: 255 if p > 0 else 0)


def _paste_crop_mask(raw: Image.Image, image_size_wh: tuple[int, int], bbox: Optional[Dict]) -> Image.Image:
    width, height = image_size_wh
    xyxy = _bbox_to_ints(bbox)
    if xyxy is None:
        resized = raw.resize((width, height), resample=Image.Resampling.NEAREST)
        return resized.point(lambda p: 255 if p > 0 else 0)
    x1, y1, x2, y2 = xyxy
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return Image.new("L", (width, height), 0)
    raw = raw.resize((x2 - x1, y2 - y1), resample=Image.Resampling.NEAREST)
    canvas = Image.new("L", (width, height), 0)
    canvas.paste(raw, (x1, y1))
    return canvas.point(lambda p: 255 if p > 0 else 0)


def load_instance_mask(mask_path: str, image_size_wh: tuple[int, int], bbox: Optional[Dict] = None) -> Image.Image:
    raw = Image.open(mask_path).convert("L")
    if raw.size == image_size_wh:
        return raw.point(lambda p: 255 if p > 0 else 0)
    if _size_matches_bbox(raw.size, bbox):
        return _paste_crop_mask(raw, image_size_wh, bbox)
    if raw.size[0] == raw.size[1]:
        return _unpad_square_canvas_mask(raw, image_size_wh)
    return _paste_crop_mask(raw, image_size_wh, bbox)


def _pick_path(record: Dict, preferred: str, fallbacks: Sequence[str]) -> str:
    fields = tuple(dict.fromkeys((preferred, *fallbacks)))
    for field in fields:
        path = record.get(field) or ""
        if path:
            return path
    return ""


def _to_clip_tensor(image: Image.Image) -> torch.Tensor:
    tensor = TF.to_tensor(image)
    return TF.normalize(tensor, mean=CLIP_MEAN, std=CLIP_STD)


def _load_image_like_original(path: str, image_size: int) -> tuple[Image.Image, tuple[int, int]]:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    height, width = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size))
    return Image.fromarray(image.astype(np.uint8)), (width, height)


def _load_clip_image(
    path: str,
    image_size: int,
    *,
    hflip: bool = False,
    rotate_angle: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if path:
        try:
            image, _ = _load_image_like_original(path, image_size)
            if hflip:
                image = TF.hflip(image)
            if rotate_angle:
                image = image.rotate(rotate_angle)
            return _to_clip_tensor(image), torch.tensor(True)
        except Exception:
            pass
    return torch.zeros(3, image_size, image_size, dtype=torch.float32), torch.tensor(False)


class JsonRefSegDataset(Dataset):
    """JSON phrase/instance segmentation dataset with optional StructXLIP side images."""

    def __init__(
        self,
        json_path: str,
        data_root: str = "",
        image_size: int = 224,
        train: bool = True,
        hflip_prob: float = 0.0,
        rotate_prob: float = 0.5,
        rotate_degrees: int = 20,
        min_similarity: float | None = None,
        use_original_caption_prefix: bool = False,
        structure_image_field: str = "filename_canny",
        chunk_top_k: int = 3,
        load_aux_images: bool = True,
        samples: Optional[List[Dict]] = None,
    ) -> None:
        self.image_size = image_size
        self.train = train
        self.hflip_prob = hflip_prob
        self.rotate_prob = rotate_prob
        self.rotate_degrees = rotate_degrees
        self.use_original_caption_prefix = use_original_caption_prefix
        self.structure_image_field = structure_image_field
        self.original_structure_image_field = "original_" + structure_image_field
        self.chunk_top_k = max(1, chunk_top_k)
        self.data_root = data_root
        self.load_aux_images = load_aux_images

        if samples is not None:
            self.samples = samples
            return

        records = _read_json(json_path)
        out: List[Dict] = []
        for item_idx, item in enumerate(records):
            image_path = _resolve(item.get("original_filename") or "", data_root)
            if not image_path:
                continue
            original_caption = item.get("original_caption", "") or ""
            original_structure_path = _resolve(_pick_path(item, self.original_structure_image_field, ORIGINAL_STRUCTURE_FALLBACK_FIELDS), data_root)
            top_segments = sorted(
                item.get("segment", []) or [],
                key=lambda x: x.get("similarity_score", float("-inf")),
                reverse=True,
            )[:self.chunk_top_k]
            edge_paths = [
                _resolve(_pick_path(seg, structure_image_field, STRUCTURE_FALLBACK_FIELDS), data_root)
                for seg in top_segments
            ]
            while len(edge_paths) < self.chunk_top_k:
                edge_paths.append("")

            for seg_idx, seg in enumerate(item.get("segment", []) or []):
                caption = (seg.get("caption") or "").strip()
                mask_path = seg.get("instance_mask") or ""
                if not caption or not mask_path:
                    continue
                score = seg.get("similarity_score", None)
                if min_similarity is not None and score is not None and float(score) < min_similarity:
                    continue
                out.append({
                    "image_path": image_path,
                    "mask_path": _resolve(mask_path, data_root),
                    "caption": caption,
                    "original_caption": original_caption,
                    "bbox": seg.get("bbox_coordinates", None),
                    "segment_image_path": _resolve(seg.get("filename") or "", data_root),
                    "structure_path": _resolve(_pick_path(seg, structure_image_field, STRUCTURE_FALLBACK_FIELDS), data_root),
                    "original_structure_path": original_structure_path,
                    "edge_paths": edge_paths,
                    "item_idx": item_idx,
                    "seg_idx": seg_idx,
                    "similarity_score": score,
                })
        if not out:
            raise ValueError(f"No valid samples found in {json_path}")
        self.samples = out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        image, original_size = _load_image_like_original(sample["image_path"], self.image_size)
        mask = load_instance_mask(sample["mask_path"], original_size, sample.get("bbox"))
        mask_arr = np.array(mask, dtype=np.uint8)
        mask_arr = cv2.resize(mask_arr, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask_arr[mask_arr < 127] = 0
        mask_arr[mask_arr >= 127] = 1
        mask = Image.fromarray(mask_arr.astype(np.uint8))

        do_hflip = self.train and self.hflip_prob > 0 and random.random() < self.hflip_prob
        rotate_angle = 0.0
        if self.train and self.rotate_prob > 0 and self.rotate_degrees > 0 and random.random() < self.rotate_prob:
            rotate_angle = random.randint(-int(self.rotate_degrees), int(self.rotate_degrees))

        if do_hflip:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if rotate_angle:
            image = image.rotate(rotate_angle)
            mask = mask.rotate(rotate_angle)

        image_tensor = _to_clip_tensor(image)
        mask_arr = np.array(mask, dtype=np.uint8)
        mask_tensor = torch.from_numpy(mask_arr).long()

        caption = sample["caption"]
        if self.use_original_caption_prefix and sample.get("original_caption"):
            text = f"In the image: {sample['original_caption']} Segment: {caption}"
        else:
            text = caption

        if self.train and self.load_aux_images:
            structure, has_structure = _load_clip_image(
                sample.get("structure_path", ""),
                self.image_size,
                hflip=do_hflip,
                rotate_angle=rotate_angle,
            )
            original_structure, has_original_structure = _load_clip_image(
                sample.get("original_structure_path", ""),
                self.image_size,
                hflip=do_hflip,
                rotate_angle=rotate_angle,
            )
            segment_image, has_segment_image = _load_clip_image(
                sample.get("segment_image_path", ""),
                self.image_size,
                hflip=do_hflip,
                rotate_angle=rotate_angle,
            )
            edge_items = [
                _load_clip_image(path, self.image_size, hflip=do_hflip, rotate_angle=rotate_angle)
                for path in sample.get("edge_paths", [])
            ]
            edge_images = torch.stack([x[0] for x in edge_items], dim=0)
            edge_valid = torch.stack([x[1] for x in edge_items], dim=0)
        else:
            zero_image = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
            structure = zero_image
            original_structure = zero_image.clone()
            segment_image = zero_image.clone()
            edge_images = torch.zeros(self.chunk_top_k, 3, self.image_size, self.image_size, dtype=torch.float32)
            has_structure = torch.tensor(False)
            has_original_structure = torch.tensor(False)
            has_segment_image = torch.tensor(False)
            edge_valid = torch.zeros(self.chunk_top_k, dtype=torch.bool)

        return {
            "image": image_tensor,
            "ground_truth_mask": mask_tensor,
            "text_prompt": text,
            "original_text": sample.get("original_caption") or text,
            "structure_image": structure,
            "original_structure_image": original_structure,
            "segment_image": segment_image,
            "edge_images": edge_images,
            "has_structure": has_structure,
            "has_original_structure": has_original_structure,
            "has_segment_image": has_segment_image,
            "edge_valid_mask": edge_valid,
            "image_name": sample["image_path"],
            "mask_name": sample["mask_path"],
            "structure_name": sample.get("structure_path", ""),
            "item_idx": sample["item_idx"],
            "seg_idx": sample["seg_idx"],
        }
