"""Download and unpack PHEME-RnR (public, figshare article 4010619) into data/pheme.

    python -m src.data.download                                   # -> data/pheme
    python -m src.data.download data/pheme path/to/pheme-rnr-dataset.tar.bz2   # use a copy you already have

figshare sometimes refuses requests that do not look like a browser (HTTP 403), so this script sends a
browser User-Agent, tries the figshare API first and then the direct file link, and finally accepts a
local copy of the archive.
"""
import json
import os
import sys
import tarfile
import time
import urllib.request

OUT = sys.argv[1] if len(sys.argv) > 1 else "data/pheme"
LOCAL = sys.argv[2] if len(sys.argv) > 2 else None
NAME = "pheme-rnr-dataset.tar.bz2"
API = "https://api.figshare.com/v2/articles/4010619"
DIRECT = ["https://ndownloader.figshare.com/files/6453753",
          "https://figshare.com/ndownloader/files/6453753"]
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/124.0 Safari/537.36",
           "Accept": "*/*"}


def fetch(url, dst=None):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120) as r:
        if dst is None:
            return r.read()
        with open(dst + ".part", "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    os.replace(dst + ".part", dst)


def is_archive(path):
    try:
        with tarfile.open(path) as tar:
            return len(tar.getmembers()) > 0
    except Exception:
        return False


def main():
    os.makedirs(OUT, exist_ok=True)
    dst = LOCAL or os.path.join(OUT, NAME)
    if not (os.path.exists(dst) and is_archive(dst)):
        urls = []
        try:
            meta = json.loads(fetch(API))
            f = next(x for x in meta["files"] if x["name"].endswith((".tar.bz2", ".tar.gz", ".tgz")))
            urls.append(f["download_url"])
            print(f"figshare lists {f['name']} ({f['size'] / 1e6:.0f} MB)")
        except Exception as e:
            print(f"figshare API not reachable ({e}); trying the direct link")
        urls += DIRECT
        for url in urls:
            for attempt in range(2):
                try:
                    print(f"Downloading {url} ...")
                    fetch(url, dst)
                    if is_archive(dst):
                        break
                    raise ValueError("downloaded file is not a valid archive")
                except Exception as e:
                    print(f"  failed: {e}")
                    time.sleep(3)
            if os.path.exists(dst) and is_archive(dst):
                break
        else:
            raise SystemExit(
                "Could not download PHEME-RnR. Download it in a browser from "
                "https://figshare.com/articles/dataset/PHEME_dataset_of_rumours_and_non-rumours/4010619 "
                f"and run: python -m src.data.download {OUT} path/to/{NAME}")
    print("Extracting ...")
    with tarfile.open(dst) as tar:
        try:
            tar.extractall(OUT, filter="data")
        except TypeError:                        # Python < 3.12
            tar.extractall(OUT)
    print("Done ->", OUT)


if __name__ == "__main__":
    main()
