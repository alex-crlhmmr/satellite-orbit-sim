#!/usr/bin/env python3
"""Download and checksum the immutable Sentinel-1 POD benchmark corpus."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--manifest", type=Path,
                        default=Path("validation/real_data/manifest.yaml"))
    args = parser.parse_args()
    manifest = yaml.safe_load(args.manifest.read_text())
    args.destination.mkdir(parents=True, exist_ok=True)
    for entry in manifest["files"]:
        target = args.destination / entry["filename"]
        if not target.exists() or sha256(target) != entry["sha256"]:
            url = manifest["dataset"]["base_url"] + entry["filename"]
            temporary = target.with_suffix(target.suffix + ".part")
            with urlopen(url) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            temporary.replace(target)
        actual = sha256(target)
        if actual != entry["sha256"]:
            raise RuntimeError(f"checksum mismatch for {target}: {actual}")
        print(f"verified {entry['split']:10s} {entry['regime']:5s} {target.name}")


if __name__ == "__main__":
    main()

