"""Download the matched METR-LA inputs published by the DCRNN authors."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import urllib.request

UPSTREAM_REVISION = "602afd9d767d3aa1c9b3eac51710d6aeee12c227"
SOURCES = {
    "metr-la.h5": "https://drive.usercontent.google.com/download?id=1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC&export=download&confirm=t",
    "adj_mx.pkl": f"https://raw.githubusercontent.com/liyaguang/DCRNN/{UPSTREAM_REVISION}/data/sensor_graph/adj_mx.pkl",
    "graph_sensor_locations.csv": f"https://raw.githubusercontent.com/liyaguang/DCRNN/{UPSTREAM_REVISION}/data/sensor_graph/graph_sensor_locations.csv",
}
EXPECTED_SHA256 = {
    "metr-la.h5": "64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05",
    "adj_mx.pkl": "a35687c6e15aa228dc45027b0ed2a0ea0f4ec78f573deb992c595092d12f61b3",
    "graph_sensor_locations.csv": "eb8ea96e07358b45d0e4ba3b89c2673fa20c54af50150249e627389e749ade6f",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, url in SOURCES.items():
        destination = out / name
        if not destination.exists():
            partial = out / (name + ".part")
            request = urllib.request.Request(url, headers={"User-Agent": "SpeedCast-LA/1.0"})
            try:
                with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as target:
                    while chunk := response.read(1024 * 1024):
                        target.write(chunk)
                with partial.open("rb") as stream:
                    signature = stream.read(8)
                if name.endswith(".h5") and signature != b"\x89HDF\r\n\x1a\n":
                    raise ValueError("Download was not an HDF5 file; check the author's Drive sharing settings")
                if sha256_file(partial) != EXPECTED_SHA256[name]:
                    raise ValueError(f"Downloaded source checksum differs from the pinned release: {name}")
                partial.replace(destination)
            except Exception:
                partial.unlink(missing_ok=True)
                raise
        digest = sha256_file(destination)
        if digest != EXPECTED_SHA256[name]:
            raise ValueError(f"Existing source checksum differs from the pinned release: {name}")
        records[name] = {
            "url": url,
            "sha256": digest,
            "bytes": destination.stat().st_size,
        }
        print(f"{name}: {records[name]['bytes']} bytes sha256={records[name]['sha256']}")
    provenance = {
        "source": "DCRNN author release",
        "repository": "https://github.com/liyaguang/DCRNN",
        "revision": UPSTREAM_REVISION,
        "download_date": datetime.now().date().isoformat(),
        "download_date_note": "Local acquisition/check date; not part of the logical data fingerprint",
        "drive_folder": "https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX",
        "files": records,
    }
    (out / "sources.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    download(args.out)


if __name__ == "__main__":
    main()
