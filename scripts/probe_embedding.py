"""Small semantic embedding smoke test, not a retrieval benchmark."""
import json
import time
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / "configs/models.json").read_text(encoding="utf-8"))["embedding"]
directory = ROOT / config["local_dir"]
tokenizer = AutoTokenizer.from_pretrained(directory, padding_side="left", local_files_only=True)
model = AutoModel.from_pretrained(directory, dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()
queries = ["上菜等了太久", "店里太吵", "价格太贵"]
documents = ["等了四十分钟还没有上菜。", "音乐很响，说话都听不清。", "这份饭要一百元，太贵了。", "交通方便，下地铁就到了。"]
texts = [f"Instruct: Given a customer complaint, retrieve reviews describing the same issue.\nQuery:{q}" for q in queries] + documents
inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
started = time.perf_counter()
with torch.inference_mode():
    vectors = model(**inputs).last_hidden_state[:, -1]
    vectors = torch.nn.functional.normalize(vectors.float(), dim=-1)
torch.cuda.synchronize()
scores = vectors[:len(queries)] @ vectors[len(queries):].T
report = {"model": config, "purpose": "synthetic smoke test only", "shape": list(vectors.shape), "all_finite": bool(torch.isfinite(vectors).all()), "top1_indices": scores.argmax(dim=1).tolist(), "expected_indices": [0, 1, 2], "cosine_scores": scores.cpu().tolist(), "seconds": time.perf_counter() - started, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
assert report["all_finite"]
report["synthetic_top1_passed"] = report["top1_indices"] == report["expected_indices"]
target = ROOT / "docs/reports/embedding-probe.json"
target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
