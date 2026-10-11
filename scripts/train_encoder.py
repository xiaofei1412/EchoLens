"""Train and choose a checkpoint on dev only; epoch-boundary resume supported."""
import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from echolens.encoder import AspectEncoder, tokenized_loader, predict
from echolens.evaluate import evaluate
from echolens.experiments import ROOT, configuration, load_split, provenance, save_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["official", "clean"], required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config, aspects = configuration()
    cfg = config["encoder"]
    output = ROOT / "artifacts/stage-01" / f"encoder-{args.version}"
    if (output / "selection.json").exists():
        raise FileExistsError("Completed run exists; preserve before rerunning")
    output.mkdir(parents=True, exist_ok=True)
    run_provenance = provenance()
    torch.set_num_threads(4)
    set_seed(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    path = ROOT / cfg["local_dir"]
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    train_ids, texts, y = load_split(args.version, "train")
    dev_ids, dev_texts, dev_y = load_split(args.version, "dev")
    assert not set(train_ids) & set(dev_ids)
    lengths = [len(t) for t in tokenizer(texts, truncation=False, verbose=False)["input_ids"]]
    train_loader = tokenized_loader(tokenizer, texts, y, cfg, shuffle=True, generator=generator)
    dev_loader = tokenized_loader(tokenizer, dev_texts, dev_y, cfg)
    model = AspectEncoder(path, len(aspects)).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    total_steps = len(train_loader) * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * cfg["warmup_ratio"]), total_steps)
    start_epoch, best_score, history = 0, -1.0, []
    checkpoint = output / "last.pt"
    if args.resume:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert state["provenance"] == run_provenance, "Resume provenance mismatch"
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        generator.set_state(state["generator"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        start_epoch, best_score, history = state["epoch"], state["best_score"], state["history"]
        del state
    elif checkpoint.exists():
        raise FileExistsError("Interrupted run exists; use --resume")
    for epoch in range(start_epoch, cfg["epochs"]):
        started = time.perf_counter()
        model.train()
        torch.cuda.reset_peak_memory_stats()
        loss_sum = 0.0
        for step, (inputs, targets) in enumerate(train_loader, 1):
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            targets = targets.to("cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**inputs)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 4), targets.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * len(targets)
            if step % 100 == 0 or step == 1:
                print(f"{args.version} epoch={epoch+1} step={step}/{len(train_loader)} loss={loss.item():.4f} elapsed={time.perf_counter()-started:.1f}s", flush=True)
        train_seconds = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated() / 2**30
        predictions, timing = predict(model, dev_loader)
        metrics = evaluate(dev_y, predictions, aspects)
        score = metrics["pairs"]["micro_f1"]
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / len(y), "train_seconds": train_seconds,
                        "training_peak_allocated_gib": peak, "dev": metrics, "dev_timing": timing})
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), output / "best.pt")
            np.save(output / "best-dev-predictions.npy", predictions)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "epoch": epoch + 1, "best_score": best_score, "history": history, "provenance": run_provenance,
                 "generator": generator.get_state(), "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}
        torch.save(state, output / "last.tmp")
        (output / "last.tmp").replace(checkpoint)
        save_json(output / "progress.json", history)
        print(f"epoch={epoch+1} dev pair micro_f1={score:.6f}", flush=True)
    del optimizer, scheduler, model
    if "state" in locals():
        del state
    torch.cuda.empty_cache()
    reloaded = AspectEncoder(path, len(aspects), pretrained=False).to("cuda")
    reloaded.load_state_dict(torch.load(output / "best.pt", weights_only=True, map_location="cpu"))
    predictions, _ = predict(reloaded, dev_loader)
    assert np.array_equal(predictions, np.load(output / "best-dev-predictions.npy")), "Checkpoint predictions changed"
    report = {"model": cfg["repo_id"], "data_version": args.version, "config": config, **run_provenance,
              "gpu": torch.cuda.get_device_name(), "selected_epoch": max(history, key=lambda h: h["dev"]["pairs"]["micro_f1"])["epoch"],
              "epochs": history, "train_rows": len(y), "dev_rows": len(dev_y), "test_read": False,
              "train_truncated_rows": sum(n > cfg["max_length"] for n in lengths),
              "train_token_length_p50_p95": np.percentile(lengths, [50, 95]).tolist(),
              "reload_predictions_equal": True, "model_sha256": sha256(output / "best.pt")}
    save_json(output / "selection.json", report)
    save_json(ROOT / "docs/reports" / f"stage1-encoder-{args.version}-dev.json", report)


if __name__ == "__main__":
    main()
