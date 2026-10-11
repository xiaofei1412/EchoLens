"""AI first-pass notes for train-only exploration; never gold annotations."""
import argparse
import json
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from probe_model import load_model

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=200)
parser.add_argument("--strict-evidence", action="store_true", help="Targeted train-only prompt repair experiment")
args = parser.parse_args()
version = "reading-v2" if args.strict_evidence else "reading-v1"
config = json.loads((ROOT / "configs/models.json").read_text(encoding="utf-8"))["generation"]
records = [json.loads(line) for line in (ROOT / "data/exploration/asap-200.jsonl").read_text(encoding="utf-8").splitlines()][:args.limit]
output = ROOT / ("data/exploration/asap-ai-notes-v2.jsonl" if args.strict_evidence else "data/exploration/asap-200-ai-notes.jsonl")
done = set()
if output.exists():
    previous = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert all(r["model_revision"] == config["revision"] and r["prompt_version"] == version for r in previous), "Resume configuration mismatch"
    done = {r["source_id"] for r in previous}
if all(r["source_id"] in done for r in records):
    print("All requested records already have notes; no GPU load needed.")
    raise SystemExit(0)
tokenizer = AutoTokenizer.from_pretrained(ROOT / config["local_dir"], local_files_only=True)
model = load_model(ROOT / config["local_dir"])
model.eval()
instruction = '阅读餐饮评论，输出JSON：{"note":"30字以内概述主要评价或问题","evidence":"支持概述的原文连续片段，不超过20字","caution":"20字以内指出需注意的语义或数据问题，无则写无"}。不输出其他内容。不得把用户猜测当作事实。'
if args.strict_evidence:
    instruction = '只选择评论中一个核心意见。先逐字复制一段原文连续片段作为evidence，再用note概括该片段支持的一件事。严禁在evidence中加省略号、拼接不同句子、插入空格或改写任何字符。用户的指控用“作者称”表达。只输出JSON：{"evidence":"原文连续片段","note":"30字内概述","caution":"20字内指出不确定性，无则写无"}。例如原文“味道很好但是上菜太慢”，有效evidence为“上菜太慢”，无效为“味道很好...上菜太慢”。'
with output.open("a", encoding="utf-8") as stream:
    for record in records:
        if record["source_id"] in done:
            continue
        started = time.perf_counter()
        inputs = tokenizer.apply_chat_template([{"role": "system", "content": instruction}, {"role": "user", "content": record["text"]}], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True).to("cuda")
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=196, use_cache=True, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        item = {"sample_index": record["sample_index"], "source_id": record["source_id"], "model_revision": config["revision"], "prompt_version": version, "review_status": "ai_only_not_gold", "output": text, "seconds": time.perf_counter() - started}
        try:
            note = json.loads(text)
            item["valid"] = isinstance(note, dict) and set(note) == {"note", "evidence", "caution"} and all(isinstance(v, str) and v.strip() for v in note.values()) and note["evidence"] in record["text"]
            item["note"] = note
        except (ValueError, TypeError):
            item["valid"] = False
        stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        stream.flush()
        print(f'sample={record["sample_index"]}/{len(records)} valid={item["valid"]} seconds={item["seconds"]:.1f}', flush=True)
