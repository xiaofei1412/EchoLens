"""Synthetic fixture for exact-span and attribution review; not model evaluation."""
import json
from echolens.experiments import ROOT
from echolens.ingest import read_records
from echolens.quote_extraction import decode_item
from echolens.workbench import dataset_path, save_dataset, atomic_json


def main():
    folder = dataset_path("ui-context-synthetic")
    if (folder/"dataset.json").exists():
        print("Existing synthetic review history preserved")
        return
    source = ROOT/"artifacts/stage-02/context-fixture.jsonl"
    text = "前菜来了，菌菇汤很好喝！但烤南瓜搭配薄荷很奇怪。"
    source.write_text(json.dumps({"id":"context-1","text":text},ensure_ascii=False)+"\n",encoding="utf-8")
    rows, _ = read_records(source,mapping={"source_record_id":"id","text_raw":"text"},source="context-synthetic",revision="v2",rights_status="synthetic")
    save_dataset(folder,rows,{"name":"上下文复核合成样例 · 非模型成绩","kind":"synthetic_fixture","analysis_config":"configs/stage2-v2.json"})
    op = decode_item({"a":"口味","s":1,"i":"praise","t":"菌菇汤","e":"菌菇汤很好喝"},text)
    rid = rows[0]["record_id"]
    (folder/"extractions").mkdir(exist_ok=True)
    atomic_json(folder/"extractions"/f"{rid}.json",{"record_id":rid,"valid":True,"fully_valid":True,"partial":False,"opinions":[op],"error":None,"seconds":0,"fixture":True})
    atomic_json(folder/"progress.json",{"status":"complete","completed":1,"total":1})
    atomic_json(folder/"topics.json",{"version":1,"topics":[],"units":[],"assignments":{},"note":"Synthetic review fixture; not a clustering result"})
    print("Synthetic exact-span review fixture prepared")


if __name__ == "__main__":
    main()
