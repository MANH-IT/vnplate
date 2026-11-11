from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import zipfile
from pathlib import Path

import gdown

THAI_FILE_ID = "1v8a8IzCK-l7c89cuT9ChozlEqEsxjymP"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/vehicles")
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "thai_vehicles.zip"
    url = f"https://drive.google.com/uc?id={THAI_FILE_ID}"
    print(f"downloading {url} -> {zip_path}")
    gdown.download(url, str(zip_path), quiet=False)
    try:
        with zipfile.ZipFile(zip_path) as handle:
            handle.extractall(root)
        print(f"extracted to {root}")
    except zipfile.BadZipFile:
        print(f"downloaded file is not a zip: {zip_path}")
    if not args.keep_zip:
        zip_path.unlink(missing_ok=True)
        print("removed zip")


if __name__ == "__main__":
    main()
