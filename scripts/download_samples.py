"""
Download two sample photos into the samples/ folder for testing.

    python scripts/download_samples.py
"""

import urllib.request
from pathlib import Path

SAMPLES = {
    "bus.jpg": "https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg",
    "zidane.jpg": "https://github.com/ultralytics/assets/releases/download/v0.0.0/zidane.jpg",
}


def main():
    folder = Path(__file__).resolve().parents[1] / "samples"
    folder.mkdir(exist_ok=True)
    for name, url in SAMPLES.items():
        target = folder / name
        if target.exists():
            print(f"{name} already downloaded")
            continue
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(url, target)
    print(f"Sample images are in {folder}")


if __name__ == "__main__":
    main()
