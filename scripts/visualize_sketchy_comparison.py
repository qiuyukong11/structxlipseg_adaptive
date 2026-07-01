import argparse
import csv
import os
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_CLIP_ROOT = Path("output/sketchy_clipseg_100percent")
DEFAULT_STRUCT_ROOT = Path("output/sketchy_structxlipseg_100percent")
DEFAULT_OUTPUT_DIR = Path("output/sketchy_clipseg_vs_structxlipseg_visualizations")
DEFAULT_RUN_CLIP = "Seg_clip_ViT-B-16_latest"
DEFAULT_RUN_STRUCT = "Seg_structxlip_ViT-B-16_latest"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = (
        ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if bold else []
    ) + [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for name in names:
        if os.path.exists(name):
            return ImageFont.truetype(name, size=size)
    return ImageFont.load_default()


def read_csv_by_name(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["name"]: row for row in csv.DictReader(f)}


def read_metric_csv(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    if not path.exists():
        return values
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("Name")
            if not name or name.startswith("__"):
                continue
            values[name] = float(row["IoU"])
    return values


def read_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return np.asarray(mask) >= 127


def binary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum(dtype=np.float64)
    union = np.logical_or(pred, gt).sum(dtype=np.float64)
    return 1.0 if union == 0 else float(inter / union)


def overlay_mask(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    base = image.convert("RGBA")
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    rgba[mask] = (*color, 115)
    overlay = Image.fromarray(rgba)
    return Image.alpha_composite(base, overlay).convert("RGB")


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_width: int,
) -> int:
    x, y = xy
    lines = wrap_text(draw, text, font, max_width)
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        draw.text((x + (max_width - (bbox[2] - bbox[0])) / 2, y), line, font=font, fill=fill)
        y += bbox[3] - bbox[1] + 4
    return y


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)

    split_lines: list[str] = []
    for line in lines:
        if draw.textbbox((0, 0), line, font=font)[2] <= max_width:
            split_lines.append(line)
            continue
        approx = max(10, int(len(line) * max_width / max(draw.textbbox((0, 0), line, font=font)[2], 1)))
        split_lines.extend(textwrap.wrap(line, width=approx) or [line])
    return split_lines


def make_panel(
    image: Image.Image,
    title: str,
    title_font: ImageFont.ImageFont,
    panel_size: int,
    title_height: int,
    border: tuple[int, int, int],
) -> Image.Image:
    panel = Image.new("RGB", (panel_size, panel_size + title_height), "white")
    draw = ImageDraw.Draw(panel)
    draw_centered_text(draw, (0, 8), title, title_font, (30, 30, 30), panel_size)
    panel.paste(image.resize((panel_size, panel_size), Image.Resampling.BICUBIC), (0, title_height))
    draw.rectangle((0, title_height, panel_size - 1, panel_size + title_height - 1), outline=border, width=2)
    return panel


def build_visual(
    image_path: Path,
    gt_path: Path,
    clip_path: Path,
    struct_path: Path,
    text_prompt: str,
    clip_iou: float,
    struct_iou: float,
    panel_size: int,
) -> Image.Image:
    title_font = load_font(22, bold=True)
    caption_font = load_font(24, bold=False)

    original = Image.open(image_path).convert("RGB")
    base = original.resize((224, 224), Image.Resampling.BICUBIC)
    gt = read_mask(gt_path, base.size)
    clip = read_mask(clip_path, base.size)
    struct = read_mask(struct_path, base.size)

    title_height = 76
    panels = [
        make_panel(base, "Original", title_font, panel_size, title_height, (180, 180, 180)),
        make_panel(overlay_mask(base, gt, (27, 158, 80)), "GT overlay", title_font, panel_size, title_height, (27, 158, 80)),
        make_panel(
            overlay_mask(base, clip, (230, 126, 34)),
            f"CLIPSeg overlay\nIoU: {clip_iou:.4f}",
            title_font,
            panel_size,
            title_height,
            (230, 126, 34),
        ),
        make_panel(
            overlay_mask(base, struct, (41, 128, 185)),
            f"StructXLIPSeg overlay\nIoU: {struct_iou:.4f}",
            title_font,
            panel_size,
            title_height,
            (41, 128, 185),
        ),
    ]

    gap = 18
    margin = 24
    content_w = panel_size * 4 + gap * 3
    tmp = Image.new("RGB", (content_w, 10), "white")
    tmp_draw = ImageDraw.Draw(tmp)
    caption = f"Text: {text_prompt}"
    caption_lines = wrap_text(tmp_draw, caption, caption_font, content_w)
    line_h = max(28, tmp_draw.textbbox((0, 0), "Ag", font=caption_font)[3] + 5)
    caption_h = line_h * len(caption_lines) + 20
    max_panel_h = max(p.height for p in panels)

    canvas = Image.new("RGB", (content_w + 2 * margin, caption_h + max_panel_h + 2 * margin), "white")
    draw = ImageDraw.Draw(canvas)
    y = margin
    for line in caption_lines:
        bbox = draw.textbbox((0, 0), line, font=caption_font)
        draw.text((margin + (content_w - (bbox[2] - bbox[0])) / 2, y), line, font=caption_font, fill=(20, 20, 20))
        y += line_h

    x = margin
    panel_y = margin + caption_h
    for panel in panels:
        canvas.paste(panel, (x, panel_y))
        x += panel_size + gap
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-root", type=Path, default=DEFAULT_CLIP_ROOT)
    parser.add_argument("--struct-root", type=Path, default=DEFAULT_STRUCT_ROOT)
    parser.add_argument("--clip-run", default=DEFAULT_RUN_CLIP)
    parser.add_argument("--struct-run", default=DEFAULT_RUN_STRUCT)
    parser.add_argument("--seed", default="seed42")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--panel-size", type=int, default=320)
    parser.add_argument("--limit", type=int, default=0, help="Only render the first N samples; 0 renders all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clip_seg_dir = args.clip_root / "seg_results" / args.seed / args.clip_run
    clip_gt_dir = args.clip_root / "gt_results" / args.seed / args.clip_run
    struct_seg_dir = args.struct_root / "seg_results" / args.seed / args.struct_run

    clip_manifest = read_csv_by_name(clip_seg_dir / "manifest.csv")
    struct_manifest = read_csv_by_name(struct_seg_dir / "manifest.csv")
    clip_metrics = read_metric_csv(clip_seg_dir / "metrics_miou_ciou.csv")
    struct_metrics = read_metric_csv(struct_seg_dir / "metrics_miou_ciou.csv")
    names = sorted(set(clip_manifest) & set(struct_manifest))
    if args.limit > 0:
        names = names[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "text_prompt", "clip_iou", "structxlip_iou", "visualization_path"],
        )
        writer.writeheader()

        for idx, name in enumerate(names, start=1):
            row = clip_manifest[name]
            image_path = Path(row["image_path"])
            gt_path = clip_gt_dir / name
            clip_path = Path(row["prediction_path"])
            struct_path = Path(struct_manifest[name]["prediction_path"])
            if not struct_path.is_absolute():
                struct_path = Path.cwd() / struct_path
            if not clip_path.is_absolute():
                clip_path = Path.cwd() / clip_path
            if not gt_path.is_absolute():
                gt_path = Path.cwd() / gt_path

            if name in clip_metrics:
                clip_iou = clip_metrics[name]
            else:
                clip_iou = binary_iou(read_mask(clip_path, (224, 224)), read_mask(gt_path, (224, 224)))
            if name in struct_metrics:
                struct_iou = struct_metrics[name]
            else:
                struct_iou = binary_iou(read_mask(struct_path, (224, 224)), read_mask(gt_path, (224, 224)))

            visual = build_visual(
                image_path=image_path,
                gt_path=gt_path,
                clip_path=clip_path,
                struct_path=struct_path,
                text_prompt=row["text_prompt"],
                clip_iou=clip_iou,
                struct_iou=struct_iou,
                panel_size=args.panel_size,
            )
            out_path = args.output_dir / name
            visual.save(out_path)
            writer.writerow(
                {
                    "name": name,
                    "text_prompt": row["text_prompt"],
                    "clip_iou": f"{clip_iou:.6f}",
                    "structxlip_iou": f"{struct_iou:.6f}",
                    "visualization_path": str(out_path),
                }
            )
            if idx % 100 == 0 or idx == len(names):
                print(f"Rendered {idx}/{len(names)}")

    print(f"Saved visualizations to: {args.output_dir}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
