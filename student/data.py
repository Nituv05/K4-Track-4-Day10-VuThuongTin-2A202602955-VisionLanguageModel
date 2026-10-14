"""Grounding labels and auditable checks for the official session splits."""
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.labcommon import CLASSES, read_jsonl
from scripts.splitting import SPLITS, dataset_version

PROMPT = (
    'Locate all road damage in the image. Use only these labels: pothole, crack, '
    'manhole. Return only a JSON list of objects with keys "label" and "bbox_2d". '
    'Each bbox_2d is [x1,y1,x2,y2], integer coordinates on the 0–1000 scale '
    'relative to image width and height, with top-left origin. '
    'Return [] when no objects are present. Do not include explanations or markdown.'
)
COLORS = {"pothole": "red", "crack": "orange", "manhole": "cyan"}


def yolo_to_box(label, yolo):
    """Convert normalized center/size to clipped integer grounding coordinates."""
    if label not in CLASSES:
        raise ValueError(f"Unknown label: {label}")
    if len(yolo) != 4 or not all(
        isinstance(v, (int, float)) and not isinstance(v, bool)
        and math.isfinite(v) and 0 <= v <= 1 for v in yolo
    ):
        raise ValueError(f"Invalid YOLO box: {yolo}")
    xc, yc, width, height = yolo
    corners = (xc - width / 2, yc - height / 2,
               xc + width / 2, yc + height / 2)
    return {"label": label,
            "bbox_2d": [max(0, min(1000, round(v * 1000))) for v in corners]}


def objects_to_answer(objects):
    boxes = [yolo_to_box(obj["label"], obj["yolo"]) for obj in objects]
    boxes.sort(key=lambda box: (box["bbox_2d"][1], box["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":"))


def box_to_pixels(box, width, height):
    return [v * extent / 1000 for v, extent in
            zip(box["bbox_2d"], (width, height, width, height))]


def yolo_to_pixels(yolo, width, height):
    """Independent original-label path, including clipping at image boundaries."""
    xc, yc, w, h = yolo
    return [max(0, min(1, v)) * extent for v, extent in zip(
        (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2),
        (width, height, width, height))]


def verify_roundtrip(record):
    """Integer rounding may move a corner by at most half a normalized unit."""
    max_error = 0.0
    for obj in record["objects"]:
        actual = box_to_pixels(yolo_to_box(obj["label"], obj["yolo"]),
                               record["width"], record["height"])
        expected = yolo_to_pixels(obj["yolo"], record["width"], record["height"])
        for a, b, extent in zip(actual, expected,
                                (record["width"], record["height"]) * 2):
            error = abs(a - b)
            if error > extent / 2000 + 1e-9:
                raise ValueError(f"Roundtrip failed: {record['image_id']}")
            max_error = max(max_error, error)
    return max_error


def inspect_splits(data_dir="data", expected_images=None):
    """Check every image, class, ID, session, label and metadata fingerprint."""
    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "splits/meta.json").read_text(encoding="utf-8"))
    records = {split: read_jsonl(data_dir / "splits" / f"{split}.jsonl")
               for split in SPLITS}
    seen_ids, session_splits, counts, image_splits = set(), {}, {}, []
    max_error = 0.0
    for split, rows in records.items():
        if not rows:
            raise ValueError(f"Empty split: {split}")
        counts[split] = {"images": len(rows), "boxes": dict.fromkeys(CLASSES, 0)}
        for row in rows:
            image_id, session = row["image_id"], row["session"]
            if image_id in seen_ids:
                raise ValueError(f"Duplicate image_id: {image_id}")
            seen_ids.add(image_id)
            if session in session_splits and session_splits[session] != split:
                raise ValueError(f"Session leakage: {session}")
            session_splits[session] = split
            with Image.open(data_dir / row["image_path"]) as im:
                if im.size != (row["width"], row["height"]):
                    raise ValueError(f"Image size mismatch: {image_id}")
                im.verify()
            max_error = max(max_error, verify_roundtrip(row))
            for obj in row["objects"]:
                counts[split]["boxes"][obj["label"]] += 1
            image_splits.append((image_id, split))
    version = dataset_version(image_splits)
    if version != meta["dataset_version"]:
        raise ValueError("dataset_version does not match actual split IDs")
    if counts != meta["counts"]:
        raise ValueError("Split counts do not match meta.json")
    if session_splits != meta["groups"]:
        raise ValueError("Session assignments do not match meta.json")
    if meta["classes"] != list(CLASSES):
        raise ValueError("Class order does not match official labels")
    if expected_images is not None:
        actual = {s: c["images"] for s, c in counts.items()}
        if actual != expected_images:
            raise ValueError(f"Unexpected dataset sizes: {actual}")
    return records, {"dataset_version": version, "counts": counts,
                     "sessions_per_split": {s: sum(v == s for v in session_splits.values())
                                            for s in SPLITS},
                     "session_overlap": False, "max_roundtrip_error_px": max_error}


def draw_label_comparison(record, data_dir="data"):
    """Side by side: original YOLO boxes and the decoded JSON boxes."""
    with Image.open(Path(data_dir) / record["image_path"]) as source:
        source = source.convert("RGB")
    w, h = source.size
    canvas = Image.new("RGB", (2 * w, h + 40), "white")
    canvas.paste(source, (0, 40))
    canvas.paste(source, (w, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), "Original YOLO (clipped)", fill="black")
    draw.text((w + 8, 6), "JSON 0-1000 -> pixels", fill="black")
    draw.text((8, 22), record["image_id"], fill="black")
    for obj in record["objects"]:
        original = yolo_to_pixels(obj["yolo"], w, h)
        converted = box_to_pixels(yolo_to_box(obj["label"], obj["yolo"]), w, h)
        for offset, coords in ((0, original), (w, converted)):
            x1, y1, x2, y2 = coords
            draw.rectangle((x1 + offset, y1 + 40, x2 + offset, y2 + 40),
                           outline=COLORS[obj["label"]], width=2)
            draw.text((x1 + offset, max(40, y1 + 28)), obj["label"],
                      fill=COLORS[obj["label"]])
    return canvas


def write_checkpoint1(data_dir="data", out_dir="work/checkpoint1", seed=42):
    """Write real-data diagnostics and six reproducible train-only illustrations."""
    records, summary = inspect_splits(
        data_dir, expected_images={"train": 1422, "val": 276, "test": 311})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = random.Random(seed).sample(records["train"], 6)
    summary["sample_image_ids"] = [r["image_id"] for r in sample]
    summary["sample_answers"] = [objects_to_answer(r["objects"]) for r in sample]
    for i, row in enumerate(sample, 1):
        draw_label_comparison(row, data_dir).save(out_dir / f"labels_{i:02d}.png")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(write_checkpoint1(), ensure_ascii=False, indent=2))
