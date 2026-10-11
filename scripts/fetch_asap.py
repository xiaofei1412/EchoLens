"""Download the pinned official ASAP release; never silently replace a file."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT / "configs/asap.json").read_text(encoding="utf-8"))
    directory = ROOT / "data/raw/asap" / config["revision"]
    old_manifest = directory / "manifest.json"
    known_hashes = {}
    if old_manifest.exists():
        known_hashes = {item["path"]: item["sha256"] for item in json.loads(old_manifest.read_text(encoding="utf-8"))["files"]}
    manifest = {"repository": config["repository"], "revision": config["revision"], "files": []}
    for name in config["files"]:
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://raw.githubusercontent.com/{config["repository"]}/{config["revision"]}/{name}'
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".partial")
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            temporary.replace(target)
        raw = target.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if name in known_hashes and digest != known_hashes[name]:
            raise ValueError(f"Local file changed since download: {name}")
        manifest["files"].append({"path": name, "url": url, "bytes": len(raw), "sha256": digest})
        print(f"ready: {name} ({len(raw)} bytes)", flush=True)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {directory / 'manifest.json'}")


if __name__ == "__main__":
    main()
