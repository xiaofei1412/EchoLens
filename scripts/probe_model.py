"""Local 4-bit inference and QLoRA infrastructure probe, not a benchmark."""
import argparse
import gc
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import bitsandbytes as bnb
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM, set_seed
from echolens.contracts import validate_opinions

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = json.loads((ROOT / "configs/labels/asap-v0.1.json").read_text(encoding="utf-8"))
SYSTEM = '提取餐饮评论中的意见。只输出JSON，格式为{"opinions":[{"aspect":"维度ID","sentiment":"positive/negative/neutral/mixed/unknown","intent":"praise/complaint/improvement_request/help_request/statement/unknown","evidence":"原文连续片段"}]}。不推断原文没有的信息。维度：' + json.dumps(TAXONOMY["aspects"], ensure_ascii=False)


def load_model(directory):
    text_config = AutoConfig.from_pretrained(directory, local_files_only=True).get_text_config()
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        directory, local_files_only=True, dtype=torch.bfloat16, device_map={"": 0},
        config=text_config, output_loading_info=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
        attn_implementation="sdpa",
    )
    if loading.get("missing_keys") or loading.get("mismatched_keys"):
        raise RuntimeError(f"Incomplete pretrained weights: {loading}")
    return model


def prompt_tokens(tokenizer, text):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True,
    ).to("cuda")


def infer(model, tokenizer, text, max_tokens):
    inputs = prompt_tokens(tokenizer, text)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        result = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    generated = result[0, inputs["input_ids"].shape[1]:]
    output = tokenizer.decode(generated, skip_special_tokens=True)
    valid, error = False, None
    try:
        validate_opinions(json.loads(output), text, TAXONOMY)
        valid = True
    except (ValueError, TypeError, KeyError) as exc:
        error = str(exc)
    return {"text": text, "output": output, "tokens": len(generated), "seconds": time.perf_counter() - started, "schema_and_substring_valid": valid, "validation_error": error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output", default="artifacts/stage-00/probe")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--pad-to-length", type=int, default=0, help="Optional synthetic memory stress test")
    parser.add_argument("--all-linear", action="store_true", help="Probe adapters on all text linear layers")
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs/models.json").read_text(encoding="utf-8"))["generation"]
    directory = ROOT / config["local_dir"]
    report = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "model": config, "purpose": "infrastructure only; synthetic examples; no held-out quality claim", "platform": platform.platform(), "python": platform.python_version(), "versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "peft", "bitsandbytes", "accelerate"]}, "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda, "steps": args.steps}

    def checkpoint():
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    checkpoint()
    set_seed(20261007)
    # Explicitly verify quantized CUDA kernels before loading the full model.
    layer = bnb.nn.Linear4bit(64, 32, quant_type="nf4", compute_dtype=torch.bfloat16).to("cuda")
    x = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    layer(x).float().square().mean().backward()
    report["nf4_kernel_finite_gradient"] = bool(torch.isfinite(x.grad).all())
    assert report["nf4_kernel_finite_gradient"]
    del layer, x
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("Loading 4-bit text model", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    model = load_model(directory)
    model.eval()
    report["model_quantized_layers"] = sum(isinstance(m, bnb.nn.Linear4bit) for m in model.modules())
    assert report["model_quantized_layers"] > 0
    samples = [json.loads(line) for line in (ROOT / "examples/probe.jsonl").read_text(encoding="utf-8").splitlines()]
    report["base_inference"] = []
    for sample in samples[:3]:
        result = infer(model, tokenizer, sample["text"], args.max_new_tokens)
        report["base_inference"].append(result)
        checkpoint()
        print(f'inference: {result["seconds"]:.1f}s valid={result["schema_and_substring_valid"]}', flush=True)
    report["inference_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    if not args.steps:
        checkpoint()
        return
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    targets = "all-linear" if args.all_linear else ["q_proj", "k_proj", "v_proj", "o_proj"]
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=targets, task_type="CAUSAL_LM"))
    report["lora_target_modules"] = targets
    report["lora_scope"] = "all text linear layers" if args.all_linear else "full-attention projection layers only; deliberately limited infrastructure probe"
    report["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    batches = []
    for sample in samples:
        prefix = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": sample["text"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        completion = json.dumps({"opinions": sample["opinions"]}, ensure_ascii=False) + tokenizer.eos_token
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        ids = tokenizer(prefix + completion, add_special_tokens=False)["input_ids"]
        assert ids[:len(prefix_ids)] == prefix_ids, "Prompt boundary changed during tokenization"
        assert len(ids) <= 1024, len(ids)
        labels = [-100] * len(prefix_ids) + ids[len(prefix_ids):]
        mask = [1] * len(ids)
        padding = max(0, args.pad_to_length - len(ids))
        ids += [tokenizer.eos_token_id] * padding
        labels += [-100] * padding
        mask += [0] * padding
        batches.append({"input_ids": torch.tensor([ids], device="cuda"), "attention_mask": torch.tensor([mask], device="cuda"), "labels": torch.tensor([labels], device="cuda")})
    report["train_sequence_lengths"] = [b["input_ids"].shape[1] for b in batches]
    report["training_config"] = {"micro_batch": 1, "gradient_accumulation": 1, "learning_rate": 0.0001, "rank": 8, "max_length": 1024, "supervision": "assistant completion only", "data": "8 synthetic infrastructure examples"}
    report["training_config"]["synthetic_padding_length"] = args.pad_to_length
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    before = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
    model.train()
    model.config.use_cache = False
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    report["losses"] = []
    for step in range(args.steps):
        loss = model(**batches[step % len(batches)]).loss
        assert torch.isfinite(loss), "Nonfinite loss"
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads), "Nonfinite gradients"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        report["losses"].append(float(loss.detach()))
        if (step + 1) % 5 == 0:
            print(f"step={step+1} loss={loss.item():.4f}", flush=True)
            checkpoint()
    torch.cuda.synchronize()
    report["training_seconds"] = time.perf_counter() - started
    report["training_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    after = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
    report["adapter_changed"] = any(not torch.equal(before[k], after[k]) for k in before)
    assert report["adapter_changed"]
    model.save_pretrained(output / "adapter")
    tokenizer.save_pretrained(output / "adapter")
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    check_inputs = prompt_tokens(tokenizer, samples[0]["text"])
    with torch.inference_mode():
        reference_logits = model(**check_inputs, logits_to_keep=1, use_cache=False).logits.float().cpu()
    del loss, grads, optimizer, model, batches, before
    gc.collect()
    torch.cuda.empty_cache()
    print("Reloading base + saved adapter", flush=True)
    reload_base = prepare_model_for_kbit_training(load_model(directory), use_gradient_checkpointing=False)
    reloaded = PeftModel.from_pretrained(reload_base, output / "adapter")
    reloaded.eval()
    restored = {k: v.detach().cpu() for k, v in get_peft_model_state_dict(reloaded).items()}
    report["adapter_weights_equal_after_reload"] = set(after) == set(restored) and all(torch.equal(after[k], restored[k]) for k in after)
    with torch.inference_mode():
        actual_logits = reloaded(**check_inputs, logits_to_keep=1, use_cache=False).logits.float().cpu()
    report["reload_logits_max_abs_error"] = float((reference_logits - actual_logits).abs().max())
    report["reload_logits_close"] = bool(torch.allclose(reference_logits, actual_logits, atol=0.1, rtol=0.01))
    assert report["adapter_weights_equal_after_reload"] and report["reload_logits_close"]
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    checkpoint()
    print(json.dumps({k: report[k] for k in ["trainable_parameters", "training_seconds", "training_peak_allocated_gib", "adapter_changed", "adapter_weights_equal_after_reload", "reload_logits_max_abs_error"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
