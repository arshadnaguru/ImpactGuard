"""Build a small, balanced sample of REAL PHEME threads (raw JSON, original folder layout).

Writes data/samples/pheme_samples/ and packs it into data/samples/pheme_samples.tar.gz (the file kept in git).
The full corpus is downloaded by src/data/download.py; the sample lets anyone inspect the data format and
run the code end to end without the download.
    python -m src.data.make_samples --src data/pheme --per-class 10
    tar -xzf data/samples/pheme_samples.tar.gz -C data/samples      # unpack it again later
"""
import argparse
import os
import random
import shutil
import tarfile

from .pheme import find_thread_dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/pheme")
    ap.add_argument("--out", default="data/samples/pheme_samples")
    ap.add_argument("--per-class", type=int, default=10, help="threads per (event, label)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dirs = find_thread_dirs(a.src)
    if not dirs:
        raise SystemExit(f"No threads under {a.src}. Run: python -m src.data.download")
    groups = {}
    for d in dirs:
        groups.setdefault((d[3], d[2]), []).append(d)
    rng, n = random.Random(a.seed), 0
    for (event, label), items in sorted(groups.items()):
        rng.shuffle(items)
        for thread_dir, _, _, _ in items[:a.per_class]:
            dst = os.path.join(a.out, event, "rumours" if label else "non-rumours", os.path.basename(thread_dir))
            shutil.copytree(thread_dir, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("._*", ".DS_Store"))
            n += 1
    tar_path = a.out.rstrip("/") + ".tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(a.out, arcname=os.path.basename(a.out.rstrip("/")))
    print(f"Copied {n} real threads into {a.out} and packed them into {tar_path}")


if __name__ == "__main__":
    main()
