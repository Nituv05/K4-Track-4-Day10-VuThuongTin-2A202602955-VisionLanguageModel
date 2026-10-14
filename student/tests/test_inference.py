import json
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts.labcommon import CLASSES
from scripts.splitting import dataset_version
from student import inference


def rows():
    return [{"image_id": str(i), "image_path": f"images/{i}.jpg", "width": 640,
             "height": 360, "objects": [{"label": "pothole", "yolo": [0.5, 0.5, 0.2, 0.2]}]}
            for i in range(5)]


def test_interrupted_run_resumes_complete_batches_and_preserves_raw(tmp_path, monkeypatch):
    model = SimpleNamespace(config=SimpleNamespace(_commit_hash="revision"))
    path = tmp_path / "e0.jsonl"
    calls = []
    def generate(model, processor, batch, *args):
        calls.append([r["image_id"] for r in batch])
        if len(calls) == 2:
            raise RuntimeError("simulated runtime disconnect")
        return ['```json\n[]\n```'] * len(batch)
    monkeypatch.setattr(inference, "generate_batch", generate)
    with pytest.raises(RuntimeError, match="disconnect"):
        inference.run_inference(model, None, rows(), tmp_path, path, "v1", "test", batch_size=2)
    assert len(inference.read_predictions(path, ["0", "1", "2", "3", "4"], allow_prefix=True)) == 2
    resumed = inference.run_inference(model, None, rows(), tmp_path, path, "v1", "test", batch_size=2)
    assert calls == [["0", "1"], ["2", "3"], ["2", "3"], ["4"]]
    assert resumed[0]["raw"] == '```json\n[]\n```'
    assert json.loads(path.with_suffix('.run.json').read_text())["status"] == "complete"
    inference.run_inference(model, None, rows(), tmp_path, path, "v1", "test", batch_size=2)
    assert len(calls) == 4  # A completed run is reused without another model call.
    with pytest.raises(ValueError, match="Resume settings changed"):
        inference.run_inference(model, None, rows(), tmp_path, path, "v1", "test",
                                batch_size=2, max_new_tokens=512)


@pytest.mark.parametrize("predictions,message", [
    ([{"image_id": "0", "raw": "[]"}, {"image_id": "0", "raw": "[]"}], "Duplicate"),
    ([{"image_id": "1", "raw": "[]"}], "IDs"),
    ([{"image_id": "0", "raw": None}], "raw string"),
])
def test_bad_prediction_files_rejected(tmp_path, predictions, message):
    path = tmp_path / "pred.jsonl"
    path.write_text(''.join(json.dumps(p) + '\n' for p in predictions))
    with pytest.raises(ValueError, match=message):
        inference.read_predictions(path, ["0"], allow_prefix=True)


def test_generator_wrong_batch_size_never_committed(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "generate_batch", lambda *args: ["[]"])
    model = SimpleNamespace(config=SimpleNamespace())
    path = tmp_path / "e0.jsonl"
    with pytest.raises(ValueError, match="exactly one"):
        inference.run_inference(model, None, rows(), tmp_path, path, "v1", "test")
    assert path.read_text() == ""


def test_diagnostics_distinguish_format_symptoms_and_keep_raw():
    predictions = [
        {"image_id": "0", "raw": "not JSON"},
        {"image_id": "1", "raw": "```json\n[]\n```"},
        {"image_id": "2", "raw": '[{"label":"car","bbox_2d":[0,0,1001,20]}]'},
        {"image_id": "3", "raw": '[{"label":"crack","bbox_2d":[1.5,1,10,10]}]'},
        {"image_id": "4", "raw": '[{"label":"pothole","bbox_2d":[400,400,600,600]}]'},
    ]
    result = inference.prediction_diagnostics(rows(), predictions)
    assert result["symptoms_per_image"] == {
        "invalid_json": 1, "empty_list": 1, "markdown_fence": 1,
        "unknown_class": 1, "outside_0_1000": 1, "noninteger_coordinates": 1,
    }
    assert result["metrics"]["parse_valid_rate"] == 0.8
    assert result["metrics"]["micro"]["tp"] == 1
    assert result["raw_examples"] == predictions


def test_official_scoring_complete_split_and_missing_id_guard(tmp_path):
    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    (data / "splits").mkdir()
    groups, counts, pairs = {}, {}, []
    for split in ("train", "val", "test"):
        Image.new("RGB", (640, 360)).save(data / f"images/{split}.jpg")
        row = {**rows()[0], "image_id": split, "session": split,
               "image_path": f"images/{split}.jpg"}
        (data / f"splits/{split}.jsonl").write_text(json.dumps(row) + '\n')
        counts[split] = {"images": 1, "boxes": {"pothole": 1, "crack": 0, "manhole": 0}}
        groups[split] = split
        pairs.append((split, split))
    version = dataset_version(pairs)
    (data / "splits/meta.json").write_text(json.dumps({
        "dataset_version": version, "groups": groups, "counts": counts, "classes": list(CLASSES)}))
    pred = tmp_path / "pred.jsonl"
    pred.write_text(json.dumps({"image_id": "test", "raw":
        '[{"label":"pothole","bbox_2d":[400,400,600,600]}]'}) + '\n')
    evidence = inference.score_predictions(pred, tmp_path / "e0.json", data)
    assert evidence["micro"]["f1"] == 1
    assert evidence["missing_predictions"] == 0
    assert evidence["dataset_version"] == version
    assert evidence["split"] == "test"
    pred.write_text("")
    with pytest.raises(ValueError, match="IDs"):
        inference.score_predictions(pred, tmp_path / "missing.json", data)
    assert not (tmp_path / "missing.json").exists()


def test_audit_counts_logical_nf4_shape_and_rejects_quantized_vision():
    class Parameter:
        dtype = "torch.float16"
        quant_state = None
        def numel(self):
            return 6
        def is_floating_point(self):
            return True
    packed = Parameter()
    packed.quant_state = SimpleNamespace(shape=(10, 20))
    Linear4bit = type("Linear4bit", (), {})
    model = SimpleNamespace(
        config=SimpleNamespace(_commit_hash="rev"),
        named_parameters=lambda: iter([
            ("model.visual.merger.weight", Parameter()),
            ("model.language_model.layers.0.self_attn.q_proj.weight", packed),
            ("lm_head.weight", Parameter())]),
        named_modules=lambda: iter([("model.language_model.layers.0.self_attn.q_proj", Linear4bit())]))
    audit = inference.audit_model(model)
    assert audit["logical_parameters"]["total"] == 212
    assert audit["logical_parameters"]["projector"] == 6
    model.named_modules = lambda: iter([("model.visual.merger.linear_fc1", Linear4bit())])
    with pytest.raises(ValueError, match="were quantized"):
        inference.audit_model(model)
