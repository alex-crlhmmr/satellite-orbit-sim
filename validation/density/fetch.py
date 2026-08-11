#!/usr/bin/env python3
"""Download and verify the frozen GRACE-FO density corpus."""

import argparse
from pathlib import Path
import sys
from urllib.request import urlopen

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.real_data.fetch import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--manifest", type=Path,
                        default=Path("validation/density/manifest.yaml"))
    args = parser.parse_args()
    manifest = yaml.safe_load(args.manifest.read_text())
    args.destination.mkdir(parents=True, exist_ok=True)
    entries = [(entry, manifest["dataset"]["base_url"] + entry["filename"])
               for entry in manifest["files"]]
    entries.append((manifest["space_weather"], manifest["space_weather"]["url"]))
    for entry, url in entries:
        target = args.destination / entry["filename"]
        if not target.exists() or sha256(target) != entry["sha256"]:
            temporary = target.with_suffix(target.suffix + ".part")
            with (urlopen(url) as response,
                  temporary.open("wb") as output):
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            temporary.replace(target)
        if sha256(target) != entry["sha256"]:
            raise RuntimeError(f"checksum mismatch: {target}")
        print(f"verified {target.name}")


if __name__ == "__main__":
    main()
