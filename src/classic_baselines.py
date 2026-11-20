"""Non-neural baselines on the same split: majority class, and TF-IDF + logistic regression on text.

Also the simple baselines for the impact (final cascade size) task.
    python -m src.classic_baselines
"""
import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression

from src.data import load_threads, split_indices, DEADLINES
from src.train import get_args
from src.utils import best_threshold, cls_metrics


def text_upto(th, deadline):
    k = len(th["texts"]) if deadline is None else max(1, int(np.searchsorted(th["t"], deadline, side="right")))
    return " ".join(th["texts"][:k]), k


def main():
    a = get_args()
    threads = load_threads(a.data, a.max_threads, a.data_seed)
    tr, va, te = split_indices(threads, a.split, a.data_seed, a.test_event)
    y_tr = np.array([threads[i]["label"] for i in tr]); y_te = np.array([threads[i]["label"] for i in te])
    views = [("full", None)] + [(f"{d}min", d) for d in DEADLINES]

    # 1) majority class
    maj = float(y_tr.mean() >= 0.5)
    rep_maj = {v: cls_metrics(y_te, np.full(len(te), maj)) for v, _ in views}

    # 2) TF-IDF + logistic regression (bag of words over the tweets seen so far; no graph, no time)
    vec = TfidfVectorizer(max_features=5000, stop_words="english", min_df=2)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(vec.fit_transform([text_upto(threads[i], None)[0] for i in tr]), y_tr)
    y_va = np.array([threads[i]["label"] for i in va])
    thr = best_threshold(y_va, clf.predict_proba(vec.transform([text_upto(threads[i], None)[0] for i in va]))[:, 1])
    rep_lr = {v: cls_metrics(y_te, clf.predict_proba(vec.transform([text_upto(threads[i], d)[0] for i in te]))[:, 1], thr)
              for v, d in views}

    # 3) impact baselines: predict log final size from (a) the train mean, (b) the observed size so far
    size = lambda ids: np.log1p([len(threads[i]["texts"]) for i in ids])
    for v, d in views[1:]:
        obs = lambda ids: np.log1p([[text_upto(threads[i], d)[1]] for i in ids])
        lin = LinearRegression().fit(obs(tr), size(tr))
        rep_maj[v]["size_mae"] = float(np.mean(np.abs(size(te) - size(tr).mean())))
        rep_lr[v]["size_mae"] = float(np.mean(np.abs(size(te) - lin.predict(obs(te)))))

    for name, model, rep in [("majority", "majority", rep_maj), ("tfidf_logreg", "tfidf_logreg", rep_lr)]:
        out = os.path.join(a.out, f"{name}_seed{a.data_seed}"); os.makedirs(out, exist_ok=True)
        json.dump(dict(name=name, model=model, prefix_aug=False, seed=a.data_seed, n_test=len(te), test=rep),
                  open(os.path.join(out, "metrics.json"), "w"), indent=1)
        print(f"{name:>13}: " + " | ".join(f"{v} AUROC {m['auroc']:.4f}" for v, m in rep.items()))
    print("Impact baselines use: majority row = train-mean size, tfidf_logreg row = linear fit on observed size.")


if __name__ == "__main__":
    main()
