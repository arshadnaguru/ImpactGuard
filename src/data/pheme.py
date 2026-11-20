"""PHEME-RnR loading, splitting, featurizing and time-truncation of cascades.

Each conversation thread is one cascade graph:
    nodes  = tweets (source tweet first, then replies in time order)
    edges  = reply-to links (child -> parent), rebuilt from in_reply_to_status_id
    t      = minutes since the source tweet (this is what the temporal model uses)
    label  = 1 rumour, 0 non-rumour
"""
import json
import os
import pickle
import random
from datetime import datetime

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

TIME_FMT = "%a %b %d %H:%M:%S %z %Y"
DEADLINES = [10, 30, 60, 120]          # early-detection windows, minutes after the source tweet
MAX_NODES = 300                        # cap very long threads (keeps the earliest tweets)


def _load_json(path):
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return json.load(f)
    except Exception:
        return None


def _json_files(d):
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".json") and not f.startswith(".")]


def _tweet(js):
    """Keep only the fields we use from one raw tweet JSON."""
    u = js.get("user") or {}
    try:
        ts = datetime.strptime(js["created_at"], TIME_FMT).timestamp()
    except Exception:
        ts = None
    return dict(id=str(js.get("id_str") or js.get("id")), text=js.get("text") or "",
                parent=str(js.get("in_reply_to_status_id_str") or js.get("in_reply_to_status_id") or ""),
                ts=ts, followers=u.get("followers_count") or 0, friends=u.get("friends_count") or 0,
                verified=1.0 if u.get("verified") else 0.0)


def find_thread_dirs(root):
    """Cheap directory scan: every folder that has a `source-tweet` (or `source-tweets`) subfolder."""
    out = []
    for dirpath, dirnames, _ in os.walk(root):
        src = [d for d in dirnames if d in ("source-tweet", "source-tweets")]
        if not src:
            continue
        parts = dirpath.replace("\\", "/").split("/")
        if "non-rumours" in parts:
            label, k = 0, parts.index("non-rumours")
        elif "rumours" in parts:
            label, k = 1, parts.index("rumours")
        else:
            continue
        event = parts[k - 1].replace("-all-rnr-threads", "")
        out.append((dirpath, src[0], label, event))
        dirnames[:] = []                                  # do not descend further
    return sorted(out)


def _parse_thread(thread_dir, src_name, label, event):
    src_files = _json_files(os.path.join(thread_dir, src_name))
    if not src_files:
        return None
    src = _load_json(src_files[0])
    if not src or "created_at" not in src:
        return None
    src = _tweet(src)
    if src["ts"] is None:
        return None
    replies = []
    for p in _json_files(os.path.join(thread_dir, "reactions")):
        js = _load_json(p)
        if js and "created_at" in js:
            tw = _tweet(js)
            if tw["ts"] is not None and tw["id"] != src["id"]:
                replies.append(tw)
    replies.sort(key=lambda r: r["ts"])
    replies = replies[:MAX_NODES - 1]
    tweets = [src] + replies
    pos = {tw["id"]: i for i, tw in enumerate(tweets)}
    parent, depth, t = [-1], [0], [0.0]
    for i, tw in enumerate(tweets[1:], start=1):
        p = pos.get(tw["parent"], 0)
        if p >= i:                                        # parent must be earlier; fall back to the source
            p = 0
        parent.append(p)
        depth.append(depth[p] + 1)
        t.append(max(0.0, (tw["ts"] - src["ts"]) / 60.0))
    t = list(np.maximum.accumulate(t))                    # keep times non-decreasing
    user = [[np.log1p(tw["followers"]), np.log1p(tw["friends"]), tw["verified"]] for tw in tweets]
    return dict(id=os.path.basename(thread_dir), event=event, label=label,
                texts=[tw["text"] for tw in tweets], user=np.array(user, dtype=np.float32),
                parent=np.array(parent), depth=np.array(depth), t=np.array(t, dtype=np.float32))


def load_threads(root, max_threads=0, seed=42, cache=True):
    """Find all threads, optionally subsample BEFORE parsing (fast), parse, and cache the result."""
    cache_path = os.path.join(root, f"_cache_threads_{max_threads}_{seed}.pkl")
    if cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    dirs = find_thread_dirs(root)
    if max_threads and len(dirs) > max_threads:
        random.Random(seed).shuffle(dirs)
        dirs = sorted(dirs[:max_threads])
    threads = [th for th in (_parse_thread(*d) for d in dirs) if th is not None]
    if cache and threads:
        with open(cache_path, "wb") as f:
            pickle.dump(threads, f)
    return threads


def split_indices(threads, mode="stratified", seed=42, test_event=None):
    """70/15/15 stratified by label, or leave-one-event-out (test = one whole event)."""
    rng = random.Random(seed)
    if mode == "event":
        events = sorted({th["event"] for th in threads})
        test_event = test_event or events[-1]
        te = [i for i, th in enumerate(threads) if th["event"] == test_event]
        rest = [i for i, th in enumerate(threads) if th["event"] != test_event]
        rng.shuffle(rest)
        n_val = max(1, int(0.15 * len(rest)))
        return rest[n_val:], rest[:n_val], te
    tr, va, te = [], [], []
    for lab in (0, 1):
        ids = [i for i, th in enumerate(threads) if th["label"] == lab]
        rng.shuffle(ids)
        a, b = int(0.70 * len(ids)), int(0.85 * len(ids))
        tr += ids[:a]; va += ids[a:b]; te += ids[b:]
    return tr, va, te


class Featurizer:
    """TF-IDF -> truncated SVD text features (fitted on TRAIN tweets only) + 3 user features = 67-d nodes.

    Saved next to every checkpoint as featurizer.pkl, so a trained model can score new threads.
    """
    def __init__(self, text_dim=64, seed=42):
        self.text_dim, self.seed = text_dim, seed

    def fit(self, threads):
        texts = [tx for th in threads for tx in th["texts"]]
        self.tfidf = TfidfVectorizer(max_features=5000, stop_words="english", min_df=2).fit(texts)
        X = self.tfidf.transform(texts)
        self.k = max(2, min(self.text_dim, X.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=self.k, random_state=self.seed).fit(X)
        return self

    def transform(self, threads):
        graphs = []
        for th in threads:
            txt = self.svd.transform(self.tfidf.transform(th["texts"])).astype(np.float32)
            if self.k < self.text_dim:
                txt = np.pad(txt, ((0, 0), (0, self.text_dim - self.k)))
            graphs.append(dict(x=np.concatenate([txt, th["user"]], 1), t=th["t"], parent=th["parent"],
                               depth=th["depth"], y=th["label"], n_full=len(th["texts"]),
                               event=th["event"], id=th["id"], source_text=th["texts"][0]))
        return graphs

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


def featurize(threads, train_idx, text_dim=64, seed=42, return_featurizer=False):
    """Fit the featurizer on the training threads only (no test leakage) and featurise every thread."""
    fz = Featurizer(text_dim, seed).fit([threads[i] for i in train_idx])
    graphs = fz.transform(threads)
    return (graphs, fz) if return_featurizer else graphs


def truncate(g, deadline=None):
    """What the platform has seen `deadline` minutes after the source tweet (None = whole cascade).

    Nodes are time-sorted and every parent is earlier than its child, so a prefix is a valid graph.
    """
    k = g["n_full"] if deadline is None else max(1, int(np.searchsorted(g["t"], deadline, side="right")))
    out = dict(g)
    out.update(x=g["x"][:k], t=g["t"][:k], parent=g["parent"][:k], depth=g["depth"][:k],
               deadline=float(g["t"][-1]) if deadline is None else float(deadline))
    return out
