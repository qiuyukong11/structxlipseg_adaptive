from __future__ import annotations

import json
from pathlib import Path


JSON_PATH = Path("/mnt/data/zruan/kqy/pami/segmentation/Sketchy_test_instance_new.json")


def main() -> None:
    with JSON_PATH.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    changed = 0
    renamed = 0
    for item in records:
        for seg in item.get("segment", []) or []:
            ann_ids = seg.get("annotation_ids")
            if seg.get("caption") != "a shoe" or not isinstance(ann_ids, list) or len(ann_ids) <= 1:
                continue

            seg["caption"] = "shoes"
            mask_path = Path(seg["instance_mask"])
            if "_a_shoe_" in mask_path.name:
                new_path = mask_path.with_name(mask_path.name.replace("_a_shoe_", "_shoes_", 1))
                if mask_path.exists() and new_path != mask_path:
                    mask_path.rename(new_path)
                    renamed += 1
                seg["instance_mask"] = str(new_path)
            changed += 1

    with JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False)

    print(json.dumps({
        "json_path": str(JSON_PATH),
        "captions_changed": changed,
        "mask_files_renamed": renamed,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
