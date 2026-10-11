"""Frozen baseline/adapter generation evaluation and a 32-record QLoRA smoke run."""
import argparse
import gc
import importlib.metadata
import json
import time
from datetime import datetime, timezone

import torch
from safetensors.torch import load_file
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, prepare_model_for_kbit_training
from transformers import AutoTokenizer, set_seed
from echolens.stage3_sft import ROOT, sha256, atomic_json, read_json, encode, metrics, parse_output
from probe_model import load_model

DATA = ROOT / "data/training/stage3-sft-v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["baseline", "smoke", "evaluate"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter", help="Saved full-training adapter, required for evaluate")
    parser.add_argument("--review-data", help="Frozen independent review packet without gold answers")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    assert (args.mode == "evaluate") == bool(args.adapter), "Only evaluate requires --adapter"
    assert not args.review_data or args.mode in {"baseline", "evaluate"}
    data = ROOT / args.review_data if args.review_data else DATA
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    manifest = read_json(data / "manifest.json")
    assert all(sha256(data / f) == h for f, h in manifest["files"].items())
    config = manifest["model"]
    tokenizer = AutoTokenizer.from_pretrained(ROOT / config["local_dir"], local_files_only=True)
    prompt = (data / "system.txt").read_text(encoding="utf-8")
    aspects = read_json(data / "aspects.json")
    dev = [json.loads(line) for line in (data / ("records.jsonl" if args.review_data else "dev.jsonl")).read_text(encoding="utf-8").splitlines()]
    identity = {"mode": args.mode, "model": config, "data_manifest_sha256": sha256(data / "manifest.json"),
                "max_new_tokens": args.max_new_tokens, "max_length": args.max_length, "steps": args.steps,
                "script_sha256": sha256(__file__), "metrics_sha256": sha256(ROOT / "src/echolens/stage3_sft.py")}
    if args.adapter:
        adapter = ROOT / args.adapter
        identity["adapter"] = {"path": args.adapter, "files": {
            name: sha256(adapter / name) for name in ["adapter_config.json", "adapter_model.safetensors"]}}
        if args.review_data:
            assert identity["adapter"]["files"] == manifest["adapter_files"], "Independent review adapter is frozen"
    previous = read_json(out / "report.json")
    if previous:
        assert previous["identity"] == identity, "Run parameters/code changed; use a new output directory"
        if previous.get("completed"):
            print("Completed run preserved", flush=True)
            return
        assert args.mode in {"baseline", "evaluate"}, "Interrupted smoke requires a new run directory"
    report = {"identity": identity, "started_at": datetime.now(timezone.utc).isoformat(),
              "versions": {n: importlib.metadata.version(n) for n in ["torch", "transformers", "peft", "bitsandbytes", "accelerate"]},
              "gpu": torch.cuda.get_device_name(), "human_gold": False, "completed": False}
    def save():
        atomic_json(out / "report.json", report)
    save()
    set_seed(20261011)
    torch.cuda.reset_peak_memory_stats()
    print("Loading fixed NF4 text model", flush=True)
    model = load_model(ROOT / config["local_dir"])
    if args.adapter:
        model = PeftModel.from_pretrained(model, ROOT / args.adapter)
        saved_weights = load_file(str(ROOT / args.adapter / "adapter_model.safetensors"))
        loaded_weights = get_peft_model_state_dict(model)
        report["adapter_reload_equal"] = set(saved_weights) == set(loaded_weights) and all(
            torch.equal(saved_weights[k], loaded_weights[k].detach().cpu()) for k in saved_weights)
        assert report["adapter_reload_equal"], "Saved and loaded adapter weights differ"
        del saved_weights, loaded_weights
        save()
    model.eval()
    if args.mode in {"baseline", "evaluate"}:
        dest = out / "predictions.jsonl"
        results = [json.loads(line) for line in dest.read_text(encoding="utf-8").splitlines()] if dest.exists() else []
        done = {r["record_id"] for r in results}
        assert len(done) == len(results) and done <= {r["record_id"] for r in dev}
        for row in dev:
            if row["record_id"] in done:
                continue
            ids = encode(tokenizer, prompt, row["text"])
            inputs = torch.tensor([ids], device="cuda")
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                    max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
            tokens = generated[0, len(ids):].tolist()
            result = {"record_id": row["record_id"], "output": tokenizer.decode(tokens, skip_special_tokens=True),
                      "tokens": len(tokens), "seconds": time.perf_counter() - started,
                      "hit_output_limit": len(tokens) >= args.max_new_tokens}
            with dest.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            results.append(result)
            report["completed_records"] = len(results)
            report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            save()
            print(f"{args.mode} {len(results)}/{len(dev)} tokens={len(tokens)} seconds={result['seconds']:.1f}", flush=True)
        if args.review_data:
            by_id = {r["record_id"]: r for r in dev}
            checks = [parse_output(r["output"], by_id[r["record_id"]]["text"], aspects) for r in results]
            report["mechanical_checks"] = {"records":len(results), "json_valid":sum(c["json_valid"] for c in checks),
                "schema_valid":sum(c["schema_valid"] for c in checks), "human_reviews":0,
                "note":"No gold answers; no accuracy or F1 computed"}
        else:
            report["metrics"] = metrics(results, dev, aspects)
        times = sorted(r["seconds"] for r in results)
        report.update(mean_seconds=sum(times)/len(times), median_seconds=times[len(times)//2],
                      output_limit_records=sum(r["hit_output_limit"] for r in results))
    else:
        train = [json.loads(line) for line in (DATA / "train.jsonl").read_text(encoding="utf-8").splitlines()]
        eligible = sorted([r for r in train if r["tokens"] <= args.max_length], key=lambda r: (r["tokens"], r["record_id"]))
        assert len(eligible) >= 32
        # Evenly spaced real lengths, including the shortest and longest eligible record.
        selected = [eligible[round(i*(len(eligible)-1)/31)] for i in range(32)]
        selected = sorted(selected, key=lambda r: r["tokens"], reverse=True)
        atomic_json(out / "sample-manifest.json", {"record_ids": [r["record_id"] for r in selected],
                    "tokens": [r["tokens"] for r in selected], "excluded_over_length": [r["record_id"] for r in train if r["tokens"] > args.max_length]})
        report["training"] = {"records": 32, "rank": 8, "alpha": 16, "dropout": 0.05,
                              "target_modules": "all-linear text only", "micro_batch": 1, "gradient_accumulation": 16,
                              "learning_rate": 1e-4, "max_length": args.max_length,
                              "autocast": "bfloat16",
                              "excluded_over_length": len(train)-len(eligible), "supervision": "assistant only",
                              "sequence_lengths": [r["tokens"] for r in selected]}
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"))
        model.config.use_cache = False
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
        before = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
        report["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        encoded = [encode(tokenizer, prompt, r["text"], r["answer"]) for r in selected]
        def batch(i):
            ids, labels = encoded[i]
            return {"input_ids": torch.tensor([ids], device="cuda"), "labels": torch.tensor([labels], device="cuda"),
                    "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device="cuda")}
        model.train()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        losses = []
        save()
        try:
            for step in range(args.steps):
                total = 0.0
                for micro in range(16):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = model(**batch((step*16+micro) % 32)).loss
                    assert torch.isfinite(loss), "Nonfinite loss"
                    total += float(loss.detach())
                    (loss / 16).backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.param_groups[0]["lr"] = 1e-4 * min(1.0, (step+1)/max(1, round(args.steps*0.05)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(total/16)
                report.update(losses=losses, completed_steps=step+1, last_gradient_norm=float(norm),
                              peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                save()
                print(f"smoke step={step+1}/{args.steps} loss={losses[-1]:.4f} peak={report['peak_allocated_gib']:.2f}GiB", flush=True)
        except Exception as exc:
            report.update(error=repr(exc), peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            save()
            raise
        torch.cuda.synchronize()
        report["training_seconds"] = time.perf_counter()-started
        after = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
        report["adapter_changed"] = any(not torch.equal(before[k], after[k]) for k in before)
        assert report["adapter_changed"]
        model.save_pretrained(out / "adapter")
        tokenizer.save_pretrained(out / "adapter")
        torch.save({"optimizer": optimizer.state_dict(), "steps": args.steps, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()}, out / "optimizer.pt")
        model.eval()
        model.gradient_checkpointing_disable()
        check = torch.tensor([encode(tokenizer, prompt, selected[-1]["text"])], device="cuda")
        with torch.inference_mode():
            expected = model(input_ids=check, logits_to_keep=1, use_cache=False).logits.float().cpu()
        del loss, optimizer, model, before
        gc.collect()
        torch.cuda.empty_cache()
        print("Reloading saved adapter", flush=True)
        base = prepare_model_for_kbit_training(load_model(ROOT / config["local_dir"]), use_gradient_checkpointing=False)
        restored = PeftModel.from_pretrained(base, out / "adapter").eval()
        weights = {k: v.detach().cpu() for k, v in get_peft_model_state_dict(restored).items()}
        report["adapter_reload_equal"] = set(after) == set(weights) and all(torch.equal(after[k], weights[k]) for k in after)
        with torch.inference_mode():
            actual = restored(input_ids=check, logits_to_keep=1, use_cache=False).logits.float().cpu()
        report["reload_logits_max_abs_error"] = float((expected-actual).abs().max())
        assert report["adapter_reload_equal"] and torch.allclose(expected, actual, atol=0.1, rtol=0.01)
    report.update(completed=True, finished_at=datetime.now(timezone.utc).isoformat())
    save()
    print("Run complete", flush=True)


if __name__ == "__main__":
    main()
