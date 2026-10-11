"""Separate analysis datasets keep baseline results and human history intact."""
import shutil

from echolens.candidate import CONFIG
from echolens.workbench import dataset_path, read_json, records_at, save_dataset, job_running


def create_candidate(source_id):
    source = dataset_path(source_id)
    metadata = read_json(source/'dataset.json')
    if metadata.get('analysis_config') == CONFIG:
        return source_id
    if job_running(source):
        raise ValueError('请等待当前数据分析完成后创建候选版本')
    target_id = source_id+'-qlora-v1'
    target = dataset_path(target_id)
    existing = read_json(target/'dataset.json')
    if existing:
        if existing.get('source_snapshot_sha256') != metadata['snapshot_sha256']:
            raise ValueError('原数据已变化，请使用新的来源版本')
        return target_id
    save_dataset(target,records_at(source),{'name':metadata['name']+' · QLoRA候选',
        'kind':metadata['kind'],'analysis_config':CONFIG,'baseline_dataset':source_id,
        'source_snapshot_sha256':metadata['snapshot_sha256'],
        'note':'Independent candidate run; no copied review decisions or baseline topics'})
    if (source/'classifications.json').exists():
        shutil.copyfile(source/'classifications.json',target/'classifications.json')
    return target_id
