"""Fetch only the pinned encoder files and record local file hashes."""
from huggingface_hub import snapshot_download
from echolens.experiments import ROOT, configuration, sha256, save_json


def main():
    config, _ = configuration()
    model = config["encoder"]
    path = ROOT / model["local_dir"]
    snapshot_download(model["repo_id"], revision=model["revision"], local_dir=path,
                      allow_patterns=["*.json", "*.txt", "*.md", "pytorch_model.bin"], max_workers=2)
    files = {p.name: sha256(p) for p in sorted(path.iterdir()) if p.is_file()}
    save_json(ROOT / "docs/reports/stage1-encoder-snapshot.json", {"repo_id": model["repo_id"], "revision": model["revision"], "files_sha256": files})
    print(f"Verified {len(files)} encoder files")


if __name__ == "__main__":
    main()
