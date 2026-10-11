"""Clearly marked synthetic UI fixture; never part of model evaluation."""
import json
from echolens.experiments import ROOT
from echolens.ingest import read_records
from echolens.workbench import dataset_path, save_dataset, atomic_json, decode_opinions, ASPECTS


def main():
    source = ROOT / "artifacts/stage-02/ui-fixture.jsonl"
    source.parent.mkdir(parents=True, exist_ok=True)
    texts = ["味道很好。上菜太慢，希望提前告知等待时间。", "服务员很耐心。等菜时间太长。", "店里太吵。音乐可以调低一些。", "环境很安静。适合聊天。"]
    source.write_text("".join(json.dumps({"id": str(i+1), "text": t}, ensure_ascii=False)+"\n" for i,t in enumerate(texts)), encoding="utf-8")
    records, _ = read_records(source, mapping={"source_record_id":"id","text_raw":"text"}, source="ui-synthetic", revision="v1", rights_status="synthetic")
    folder = dataset_path("ui-synthetic")
    save_dataset(folder, records, {"name":"合成验收样例 · 非模型成绩", "kind":"synthetic_fixture"})
    (folder / "extractions").mkdir(exist_ok=True)
    classifications, units, assignments = {}, [], {}
    for row in records:
        index = int(row["source_record_id"])-1
        a = 6 if index < 2 else 11
        e = 1 if index < 2 else 0
        s = 1 if index == 3 else -1
        payload = {"opinions":[{"a":a,"s":s,"i":"praise" if s==1 else "complaint","e":e}]}
        opinions = decode_opinions(json.dumps(payload), row["text_raw"])
        rid = row["record_id"]
        atomic_json(folder / "extractions" / f"{rid}.json", {"record_id":rid,"valid":True,"opinions":opinions,"output":json.dumps(payload),"error":None,"seconds":0,"fixture":True})
        labels = [-2]*len(ASPECTS)
        labels[a] = s
        classifications[rid] = {"labels":labels,"confidence":[.8]*len(ASPECTS)}
        unit = {"unit_id":rid+":0","record_id":rid,"centrality":.8,**opinions[0]}
        units.append(unit)
        assignments[unit["unit_id"]] = str(index//2)
    atomic_json(folder / "classifications.json", classifications)
    if not (folder / "topics.json").exists():
        atomic_json(folder / "topics.json", {"version":1,"topics":[{"id":"0","name":"合成 · 等菜反馈","keywords":["等菜"],"origin":"synthetic_fixture"},{"id":"1","name":"合成 · 噪音反馈","keywords":["噪音"],"origin":"synthetic_fixture"}],"units":units,"assignments":assignments})
    atomic_json(folder / "progress.json", {"status":"complete","completed":4,"total":4})
    print("Synthetic UI fixture prepared; presets are not model results")


if __name__ == "__main__":
    main()
