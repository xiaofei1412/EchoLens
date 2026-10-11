"""Batch short contextual windows and preserve rejected opinions alongside accepted ones."""
import json
import os
import time
import torch
from echolens.experiments import ROOT
from echolens.extract import load_extractor
from echolens.opinions_v2 import clauses, merge_adjacent
from echolens.quote_extraction import messages, decode_item
from echolens.workbench import atomic_json, read_json


def windows(text, chars=180, context=3):
    spans = clauses(text)
    result, first = [], 0
    while first < len(spans):
        last = first
        while last+1 < len(spans) and spans[last+1]["end"]-spans[first]["start"] <= chars:
            last += 1
        begin = max(0, first-context)
        start, end = spans[begin]["start"], spans[min(last+context,len(spans)-1)]["end"]
        result.append({"text":text[start:end],"offset":start,"clause_offset":begin,"core_start":spans[first]["start"],"core_end":spans[last]["end"]})
        first = last+1
    return result


def decode_window(output, text, limit):
    payload = json.loads(output)
    if not isinstance(payload,dict) or set(payload) != {"opinions"} or not isinstance(payload["opinions"],list):
        raise ValueError("输出必须是仅包含 opinions 数组的 JSON")
    accepted, rejected = [], []
    if len(payload["opinions"]) > limit:
        rejected.append({"error":"意见数量超过提示上限，已逐条校验并保留，需复核"})
    for i, item in enumerate(payload["opinions"]):
        try:
            opinion = decode_item(item,text)
            opinion["raw_indices"] = [i]
            accepted.append(opinion)
        except (ValueError,TypeError,KeyError) as exc:
            rejected.append({"index":i,"raw_opinion":item,"error":str(exc)})
    return accepted, rejected


def run_windows(folder, rows, total, config, models):
    started = time.perf_counter()
    work = []
    for row in rows:
        if (folder/"extractions"/f"{row['record_id']}.json").exists():
            continue
        for i, window in enumerate(windows(row["text_raw"],config["window_chars"],config["context_clauses"])):
            work.append({"record_id":row["record_id"],"index":i,**window})
    (folder/"windows").mkdir(exist_ok=True)
    pending = [w for w in work if not (folder/"windows"/f"{w['record_id']}-{w['index']}.json").exists()]
    if pending:
        tokenizer, model = load_extractor(ROOT/models["generation"]["local_dir"],quantized=config.get("quantized",True))
        torch.cuda.reset_peak_memory_stats()
        for index in range(0,len(pending),config["generation_batch_size"]):
            batch = pending[index:index+config["generation_batch_size"]]
            prompts = [tokenizer.apply_chat_template(messages(w["text"]),tokenize=False,add_generation_prompt=True,enable_thinking=False) for w in batch]
            inputs = tokenizer(prompts,padding=True,add_special_tokens=False,return_tensors="pt").to("cuda")
            if inputs["input_ids"].shape[1] > config["max_input_tokens"]:
                raise ValueError("窗口输入超过长度上限，未静默截断")
            batch_start = time.perf_counter()
            with torch.inference_mode():
                outputs = model.generate(**inputs,max_new_tokens=config["max_new_tokens"],do_sample=False,pad_token_id=tokenizer.pad_token_id)
            torch.cuda.synchronize()
            seconds = time.perf_counter()-batch_start
            for window, output in zip(batch,outputs[:,inputs["input_ids"].shape[1]:]):
                raw = tokenizer.decode(output,skip_special_tokens=True).strip()
                result = {"output":raw,"seconds":seconds/len(batch),"generation_batch_size":len(batch),"opinions":[],"rejected":[],"error":None}
                try:
                    result["opinions"],result["rejected"] = decode_window(raw,window["text"],config["max_opinions"])
                except (ValueError,TypeError,KeyError) as exc:
                    result["error"] = str(exc)
                atomic_json(folder/"windows"/f"{window['record_id']}-{window['index']}.json",result)
            atomic_json(folder/"progress.json",{"status":"running","pid":os.getpid(),"phase":"context_windows","completed_windows":len(work)-len(pending)+index+len(batch),"total_windows":len(work),"total":total,"peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30})
            print(f"Windows {index+len(batch)}/{len(pending)} elapsed={time.perf_counter()-started:.1f}s peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB",flush=True)
    for row in rows:
        rid = row["record_id"]
        path = folder/"extractions"/f"{rid}.json"
        if path.exists():
            continue
        opinions, rejects, outputs, seconds = [], [], [], 0
        for i, window in enumerate(windows(row["text_raw"],config["window_chars"],config["context_clauses"])):
            result = read_json(folder/"windows"/f"{rid}-{i}.json")
            # Revalidate cached raw output with the recorded postprocessing version.
            result.update(opinions=[], rejected=[], error=None)
            try:
                result["opinions"],result["rejected"] = decode_window(result["output"],window["text"],config["max_opinions"])
            except (ValueError,TypeError,KeyError) as exc:
                result["error"] = str(exc)
            seconds += result["seconds"]
            outputs.append(result["output"])
            rejects.extend({"window":i,**r} for r in result["rejected"])
            if result["error"]:
                rejects.append({"window":i,"error":result["error"]})
            for op in result["opinions"]:
                op["start"] += window["offset"]
                op["end"] += window["offset"]
                op["sentence_start"] += window["clause_offset"]
                op["sentence_end"] += window["clause_offset"]
                op["raw_indices"] = [f"{i}:{j}" for j in op["raw_indices"]]
                if op["end"] > window["core_start"] and op["start"] < window["core_end"]:
                    opinions.append(op)
        opinions = merge_adjacent(opinions,row["text_raw"])
        flags = [flag for op in opinions for flag in op.get("review_flags",[])]
        fully_valid = not rejects and not flags
        atomic_json(path,{"record_id":rid,"valid":bool(opinions) or fully_valid,"fully_valid":fully_valid,
                          "partial":bool(opinions) and not fully_valid,"opinions":opinions,"rejected":rejects,
                          "review_flags":flags,
                          "error":None if fully_valid else "部分意见未通过校验" if opinions else "窗口意见未通过校验",
                          "seconds":seconds,"output":"\n\n".join(outputs)})
    done = len(list((folder/"extractions").glob("*.json")))
    atomic_json(folder/"progress.json",{"status":"complete" if done==total else "pilot_complete","phase":"extraction","completed":done,"total":total,"seconds":time.perf_counter()-started})
    print(f"Context records {done}/{total}",flush=True)
