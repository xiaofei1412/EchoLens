"""Three train-pool examples for integration only, never held-out quality claims."""
from echolens.candidate_workspace import create_candidate
from echolens.workbench import dataset_path, records_at, save_dataset


def main():
    source = dataset_path('asap-demo-1000-v3')
    folder = dataset_path('stage3-integration3')
    if not (folder/'dataset.json').exists():
        rows = records_at(source)
        selected = [next(r for r in rows if low<=len(r['text_raw'])<=high) for low,high in [(80,150),(250,400),(550,750)]]
        save_dataset(folder,selected,{'name':'阶段3 · 3条接入验证','kind':'integration_probe',
            'analysis_config':'configs/stage2-v2.json','source':'asap-demo-1000-v3 train pool',
            'note':'Engineering only, not an independent evaluation'})
    print(create_candidate(folder.name))


if __name__=='__main__':
    main()
