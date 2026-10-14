"""Qwen3-VL zero-shot inference, resumable raw outputs and official scoring."""
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

from scripts.labcommon import read_jsonl
from scripts.score import evaluate, parse_output
from student.data import PROMPT, inspect_splits

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def audit_model(model):
    """Count logical weights, rather than the packed storage of NF4 tensors."""
    counts = dict.fromkeys(("total", "visual", "projector", "language_model", "lm_head"), 0)
    for name, param in model.named_parameters():
        state = getattr(param, "quant_state", None)
        count = math.prod(state.shape) if state is not None else param.numel()
        counts["total"] += count
        if name.startswith("model.visual."):
            counts["visual"] += count
            if name.startswith(("model.visual.merger.", "model.visual.deepstack_merger_list.")):
                counts["projector"] += count
            if param.is_floating_point() and str(param.dtype) != "torch.float16":
                raise ValueError(f"Expected fp16 vision parameter: {name} ({param.dtype})")
        elif name.startswith("model.language_model."):
            counts["language_model"] += count
        elif name.startswith("lm_head."):
            counts["lm_head"] += count
    quantized = [name for name, module in model.named_modules()
                 if type(module).__name__ == "Linear4bit"]
    if not quantized:
        raise ValueError("No NF4 linear modules found; check quantization installation")
    forbidden = [name for name in quantized
                 if name.startswith(("model.visual.", "lm_head"))]
    if forbidden:
        raise ValueError(f"Vision/lm_head were quantized: {forbidden[:5]}")
    if not counts["visual"] or not counts["language_model"] or not counts["projector"]:
        raise ValueError("Unexpected Qwen3-VL module tree; inspect model.named_modules()")
    return {"logical_parameters": counts, "quantized_linear_modules": len(quantized),
            "model_revision": getattr(model.config, "_commit_hash", None)}


def load_model_processor(model_id=MODEL_ID):
    """Lazy imports keep CPU data/tests independent of the Colab GPU stack."""
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("NF4 inference needs a CUDA GPU; run this cell on Colab T4")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["visual", "lm_head"],
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=quantization, dtype=torch.float16,
        device_map={"": 0}, attn_implementation="sdpa",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    return model, processor, audit_model(model)


def generate_batch(model, processor, rows, data_dir, prompt=PROMPT, max_new_tokens=256):
    import torch

    conversations = []
    for row in rows:
        with Image.open(Path(data_dir) / row["image_path"]) as source:
            image = source.convert("RGB")
        conversations.append([{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt}]}])
    inputs = processor.apply_chat_template(
        conversations, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", padding=True,
    )
    # Some tokenizer versions return this unrelated text-only field.
    # Keep mm_token_type_ids: it is a separate multimodal model input.
    inputs.pop("token_type_ids", None)
    image_tokens = (inputs["input_ids"] == model.config.image_token_id).sum(dim=1).tolist()
    for row, count in zip(rows, image_tokens):
        if (row["width"], row["height"]) == (640, 360) and count != 220:
            raise ValueError(f"Expected 220 image tokens for {row['image_id']}; got {count}")
    inputs = inputs.to(model.device)
    # Trim the padded input width, not each row's attention-mask length.
    prompt_width = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            num_beams=1, use_cache=True, return_dict_in_generate=False,
        )
    return processor.batch_decode(
        output[:, prompt_width:], skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def read_predictions(path, expected_ids, allow_prefix=False):
    """Reject duplicates, missing/extra IDs and changed order before resuming/scoring."""
    rows = read_jsonl(path)
    ids = [row["image_id"] for row in rows]
    expected = list(expected_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate prediction image_id")
    target = expected[:len(ids)] if allow_prefix else expected
    if ids != target:
        raise ValueError("Prediction IDs must match the selected split in order")
    if any(not isinstance(row.get("raw"), str) for row in rows):
        raise ValueError("Every prediction must contain a raw string")
    return rows


def run_inference(model, processor, rows, data_dir, pred_path, dataset_version,
                  split, model_id=MODEL_ID, batch_size=4, max_new_tokens=256,
                  prompt=PROMPT, progress_callback=None):
    """Commit each complete batch atomically; restart without regenerating saved rows."""
    if batch_size < 1 or max_new_tokens < 1 or not rows:
        raise ValueError("Need nonempty rows and positive batch_size/max_new_tokens")
    expected_ids = [row["image_id"] for row in rows]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Duplicate input image_id")
    pred_path = Path(pred_path)
    run_path = pred_path.with_suffix(".run.json")
    settings = {"inference_version": 1, "model_id": model_id,
                "model_revision": getattr(model.config, "_commit_hash", None),
                "dataset_version": dataset_version, "split": split,
                "image_ids": expected_ids, "prompt": prompt, "coord": "norm1000",
                "quantization": "nf4-4bit", "compute_dtype": "float16",
                "unquantized_modules": ["visual", "lm_head"],
                "max_new_tokens": max_new_tokens, "do_sample": False,
                "batch_size": batch_size}
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    prior = {}
    if pred_path.exists():
        if not run_path.exists():
            raise ValueError("Existing predictions lack run metadata; choose a new output path")
        prior = json.loads(run_path.read_text(encoding="utf-8"))
        if prior.get("fingerprint") != fingerprint:
            raise ValueError("Resume settings changed; choose a new output path")
        predictions = read_predictions(pred_path, expected_ids, allow_prefix=True)
    else:
        predictions = []
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    prior_seconds = prior.get("inference_seconds", 0)
    def persist():
        temporary = pred_path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                     for row in predictions), encoding="utf-8")
        temporary.replace(pred_path)
        write_json(run_path, {"settings": settings, "fingerprint": fingerprint,
                              "completed_images": len(predictions), "total_images": len(rows),
                              "status": "complete" if len(predictions) == len(rows) else "running",
                              "inference_seconds": prior_seconds + time.perf_counter() - started})
        if progress_callback:
            progress_callback(pred_path, run_path)
    # Metadata exists even if generation fails on the first batch.
    persist()
    for start in range(len(predictions), len(rows), batch_size):
        batch = rows[start:start + batch_size]
        raw = generate_batch(model, processor, batch, data_dir, prompt, max_new_tokens)
        if len(raw) != len(batch) or any(not isinstance(value, str) for value in raw):
            raise ValueError("Generation must return exactly one raw string per image")
        predictions.extend({"image_id": row["image_id"], "raw": value}
                           for row, value in zip(batch, raw))
        persist()
        print(f"{split}: {len(predictions)}/{len(rows)} images", flush=True)
    return predictions


def prediction_diagnostics(rows, predictions):
    """Keep raw output unchanged; flag symptoms without claiming causes automatically."""
    by_id = {row["image_id"]: row["raw"] for row in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {r["image_id"] for r in rows}:
        raise ValueError("Diagnostics need one prediction per selected image")
    metrics = evaluate(rows, by_id)
    symptoms = dict.fromkeys(("invalid_json", "empty_list", "markdown_fence",
                              "unknown_class", "outside_0_1000", "noninteger_coordinates"), 0)
    for raw in by_id.values():
        symptoms["markdown_fence"] += "```" in raw
        boxes = parse_output(raw)
        if boxes is None:
            symptoms["invalid_json"] += 1
            continue
        symptoms["empty_list"] += not boxes
        symptoms["unknown_class"] += any(b["label"] not in ("pothole", "crack", "manhole") for b in boxes)
        symptoms["outside_0_1000"] += any(v < 0 or v > 1000 for b in boxes for v in b["bbox"])
        symptoms["noninteger_coordinates"] += any(not v.is_integer() for b in boxes for v in b["bbox"])
    return {"metrics": metrics, "symptoms_per_image": symptoms,
            "raw_examples": predictions[:5],
            "note": "Symptoms can overlap. Hallucination, wrong class and wrong scale need image/GT inspection."}


def score_predictions(pred_path, out_path, data_dir="data", split="test", name="E0", meta=None):
    """Use the instructor CLI, allowing only complete official splits as evidence."""
    records, summary = inspect_splits(data_dir)
    if split not in ("val", "test"):
        raise ValueError("Only val/test can be scored")
    read_predictions(pred_path, [row["image_id"] for row in records[split]])
    scorer = Path(__file__).resolve().parents[1] / "scripts/score.py"
    command = [sys.executable, str(scorer), "--pred", str(pred_path),
               "--splits-dir", str(Path(data_dir) / "splits"), "--split", split,
               "--coord", "norm1000", "--name", name, "--out", str(out_path)]
    for key, value in (meta or {}).items():
        command.extend(["--meta", f"{key}={value}"])
    subprocess.run(command, check=True)
    evidence = json.loads(Path(out_path).read_text(encoding="utf-8"))
    if evidence["missing_predictions"] or evidence["dataset_version"] != summary["dataset_version"]:
        raise ValueError("Scored evidence does not match the selected dataset")
    return evidence
