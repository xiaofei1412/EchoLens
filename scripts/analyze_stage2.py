"""Resumable local classification and few-shot extraction worker."""
import argparse
import gc
import hashlib
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import torch
from transformers import AutoTokenizer, set_seed

from echolens.encoder import AspectEncoder, tokenized_loader
from echolens.experiments import ROOT, configuration, sha256
from echolens.extract import load_extractor, messages, PROMPT, FEW_SHOT
from echolens.workbench import dataset_path, records_at, read_json, atomic_json, decode_opinions, ASPECTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="asap-demo-1000")
    parser.add_argument("--limit", type=int, help="Pilot only; already completed records are preserved")
    parser.add_argument("--pilot", action="store_true", help="Use the dataset's fixed development selection")
    parser.add_argument("--classification-only", action="store_true")
    args = parser.parse_args()
    folder = dataset_path(args.dataset)
    metadata = read_json(folder / "dataset.json")
    config = read_json(ROOT / metadata.get("analysis_config", "configs/stage2.json"))
    make_messages, decode = messages, decode_opinions
    prompt, examples = PROMPT, FEW_SHOT
    if config["version"] == "stage2-v2":
        from echolens import quote_extraction
        prompt, examples = quote_extraction.PROMPT, quote_extraction.FEW_SHOT
    elif config["version"] == "stage2-v3":
        from echolens import semantic_review
        prompt, examples = semantic_review.PROMPT, semantic_review.FEW_SHOT
    elif config["version"] == "stage3-candidate-v1":
        prompt, examples = (ROOT/config['prompt_path']).read_text(encoding='utf-8'), []
        assert args.classification_only, 'Candidate extraction uses analyze_stage3.py'
    models = read_json(ROOT / "configs/models.json")
    rows = records_at(folder)
    metadata = read_json(folder / "dataset.json")
    classifier_path = ROOT / "artifacts/stage-01/encoder-clean/best.pt"
    run = {"snapshot_sha256": metadata["snapshot_sha256"], "config": config, "models": models,
           "classifier_sha256": sha256(classifier_path),
           "prompt_sha256": hashlib.sha256((prompt+json.dumps(examples, ensure_ascii=False)).encode()).hexdigest()}
    if config.get('semantic_review'):
        source = dataset_path(metadata['baseline_dataset'])
        run['source_extractions_sha256'] = hashlib.sha256(''.join(sha256(p) for p in sorted((source/'extractions').glob('*.json'))).encode()).hexdigest()
    if (folder / "analysis-run.json").exists():
        assert read_json(folder / "analysis-run.json") == run, "Dataset/model/prompt changed; create a new dataset/run"
    else:
        atomic_json(folder / "analysis-run.json", run)
    atomic_json(folder / "progress.json", {"status": "running", "pid": os.getpid(), "phase": "classification", "total": len(rows)})
    set_seed(config["seed"])
    torch.set_num_threads(4)
    started = time.perf_counter()
    if not (folder / "classifications.json").exists():
        classifier_config, _ = configuration()
        cfg = classifier_config["encoder"]
        path = ROOT / cfg["local_dir"]
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AspectEncoder(path, len(ASPECTS), pretrained=False).to("cuda").eval()
        model.load_state_dict(torch.load(classifier_path, weights_only=True, map_location="cpu"))
        texts = [r["text_normalized"] for r in rows]
        loader = tokenized_loader(tokenizer, texts, np.zeros((len(rows), len(ASPECTS)), dtype=np.int64), cfg)
        predictions, confidences = [], []
        with torch.inference_mode():
            for inputs, _ in loader:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(**{k: v.to("cuda") for k, v in inputs.items()})
                probability = logits.float().softmax(dim=-1)
                predictions.extend((probability.argmax(dim=-1).cpu().numpy()-2).tolist())
                confidences.extend(probability.max(dim=-1).values.cpu().tolist())
        atomic_json(folder / "classifications.json", {r["record_id"]: {"labels": p, "confidence": c} for r, p, c in zip(rows, predictions, confidences)})
        del model, tokenizer, loader, logits, probability, inputs
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Classified {len(rows)} records", flush=True)
    if args.classification_only:
        return
    (folder / "extractions").mkdir(exist_ok=True)
    requested = rows[:args.limit] if args.limit else rows
    if args.pilot:
        pilot_ids = set(read_json(folder / "audit-selection.json")["pilot"])
        requested = [r for r in rows if r["record_id"] in pilot_ids]
    if config.get('semantic_review'):
        from echolens.semantic_review import run_review
        run_review(folder, requested, len(rows), config, models)
        return
    if config.get("window_chars"):
        from echolens.context_worker import run_windows
        run_windows(folder, requested, len(rows), config, models)
        return
    todo = [r for r in requested if not (folder / "extractions" / f"{r['record_id']}.json").exists()]
    if todo:
        tokenizer, model = load_extractor(ROOT / models["generation"]["local_dir"])
        torch.cuda.reset_peak_memory_stats()
        batch_size = config["generation_batch_size"]
        for index in range(0, len(todo), batch_size):
            batch = todo[index:index+batch_size]
            ready, prompts = [], []
            for row in batch:
                rendered = tokenizer.apply_chat_template(make_messages(row["text_raw"]), tokenize=False, add_generation_prompt=True, enable_thinking=config.get("enable_thinking", False))
                if len(tokenizer(rendered, add_special_tokens=False)["input_ids"]) > config["max_input_tokens"]:
                    atomic_json(folder / "extractions" / f"{row['record_id']}.json", {"record_id": row["record_id"], "valid": False, "opinions": [], "error": "输入超过本轮长度上限，未静默截断", "seconds": 0})
                else:
                    ready.append(row)
                    prompts.append(rendered)
            if ready:
                inputs = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt").to("cuda")
                batch_start = time.perf_counter()
                with torch.inference_mode():
                    outputs = model.generate(**inputs, max_new_tokens=config["max_new_tokens"], do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id)
                torch.cuda.synchronize()
                seconds = time.perf_counter()-batch_start
                for row, output in zip(ready, outputs[:, inputs["input_ids"].shape[1]:]):
                    text = tokenizer.decode(output, skip_special_tokens=True).strip()
                    result = {"record_id": row["record_id"], "output": text, "seconds": seconds/len(ready), "batch_seconds": seconds, "batch_size": len(ready), "valid": False, "opinions": [], "error": None}
                    try:
                        result["opinions"] = decode(text, row["text_raw"], config["max_opinions"])
                        result["valid"] = True
                    except (ValueError, TypeError, KeyError) as exc:
                        result["error"] = str(exc)
                    atomic_json(folder / "extractions" / f"{row['record_id']}.json", result)
            done = len(list((folder / "extractions").glob("*.json")))
            atomic_json(folder / "progress.json", {"status": "running", "pid": os.getpid(), "phase": "extraction", "completed": done, "total": len(rows), "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30})
            print(f"Extracted {done}/{len(rows)} elapsed={time.perf_counter()-started:.1f}s peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)
    done = len(list((folder / "extractions").glob("*.json")))
    atomic_json(folder / "progress.json", {"status": "complete" if done == len(rows) else "pilot_complete", "phase": "extraction", "completed": done, "total": len(rows), "finished_at": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
