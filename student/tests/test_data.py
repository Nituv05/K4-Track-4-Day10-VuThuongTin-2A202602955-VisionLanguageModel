import json

import pytest
from PIL import Image

from scripts.labcommon import CLASSES
from scripts.splitting import dataset_version
from student.data import (box_to_pixels, draw_label_comparison, inspect_splits,
                          objects_to_answer, verify_roundtrip, yolo_to_box)


def test_known_geometry_and_pixel_reconstruction():
    box = yolo_to_box("pothole", [0.5, 0.5, 0.2, 0.4])
    assert box == {"label": "pothole", "bbox_2d": [400, 300, 600, 700]}
    assert box_to_pixels(box, 640, 360) == [256, 108, 384, 252]


def test_clip_round_and_empty_answer():
    assert yolo_to_box("crack", [0.01, 0.99, 0.2, 0.2])["bbox_2d"] == [0, 890, 110, 1000]
    assert yolo_to_box("manhole", [0.1234, 0.5678, 0, 0])["bbox_2d"] == [123, 568, 123, 568]
    assert objects_to_answer([]) == "[]"


def test_answer_sorted_top_to_bottom_then_left_to_right():
    answer = objects_to_answer([
        {"label": "pothole", "yolo": [0.8, 0.8, 0.1, 0.1]},
        {"label": "crack", "yolo": [0.8, 0.2, 0.1, 0.1]},
        {"label": "manhole", "yolo": [0.2, 0.2, 0.1, 0.1]},
    ])
    assert [b["label"] for b in json.loads(answer)] == ["manhole", "crack", "pothole"]
    assert " " not in answer


@pytest.mark.parametrize("yolo", [[float("nan"), 0, 0, 0], [0, 0, 0, 2],
                                   [0, 0, 0], [True, 0, 0, 0]])
def test_bad_coordinates_rejected(yolo):
    with pytest.raises(ValueError):
        yolo_to_box("crack", yolo)


def test_unknown_label_rejected():
    with pytest.raises(ValueError):
        yolo_to_box("car", [0.5, 0.5, 0.1, 0.1])


@pytest.fixture
def small_dataset(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "splits").mkdir()
    counts, groups, pairs = {}, {}, []
    for split in ("train", "val", "test"):
        Image.new("RGB", (640, 360)).save(tmp_path / "images" / f"{split}.jpg")
        row = {"image_id": split, "session": split, "image_path": f"images/{split}.jpg",
               "width": 640, "height": 360,
               "objects": [{"label": "crack", "yolo": [0.1234, 0.99, 0.2, 0.2]}]}
        (tmp_path / "splits" / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
        counts[split] = {"images": 1, "boxes": {"pothole": 0, "crack": 1, "manhole": 0}}
        groups[split] = split
        pairs.append((split, split))
    meta = {"dataset_version": dataset_version(pairs), "counts": counts,
            "groups": groups, "classes": list(CLASSES)}
    (tmp_path / "splits/meta.json").write_text(json.dumps(meta))
    return tmp_path


def test_inspection_roundtrip_and_visualization(small_dataset):
    records, summary = inspect_splits(small_dataset)
    assert summary["session_overlap"] is False
    assert summary["max_roundtrip_error_px"] <= 0.32
    assert verify_roundtrip(records["train"][0]) <= 0.32
    image = draw_label_comparison(records["train"][0], small_dataset)
    assert image.size == (1280, 400)


@pytest.mark.parametrize("field,value,message", [
    ("session", "train", "Session leakage"),
    ("image_id", "train", "Duplicate image_id"),
    ("width", 639, "Image size mismatch"),
])
def test_data_integrity_failures(small_dataset, field, value, message):
    path = small_dataset / "splits/val.jsonl"
    row = json.loads(path.read_text())
    row[field] = value
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match=message):
        inspect_splits(small_dataset)


@pytest.mark.parametrize("field,value,message", [
    ("dataset_version", "wrong", "dataset_version"),
    ("counts", {}, "counts"),
    ("groups", {}, "Session assignments"),
])
def test_stale_metadata_rejected(small_dataset, field, value, message):
    path = small_dataset / "splits/meta.json"
    meta = json.loads(path.read_text())
    meta[field] = value
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match=message):
        inspect_splits(small_dataset)
