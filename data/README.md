# Data

| Folder | What it is | In git? |
|---|---|---|
| `data/samples/pheme_samples.tar.gz` | A small set of **real** PHEME-RnR threads (10 rumour + 10 non-rumour per event), raw JSON in the original folder layout, packed into one file | yes |
| `data/samples/pheme_samples/` | The same sample unpacked (`tar -xzf data/samples/pheme_samples.tar.gz -C data/samples`, done automatically by `scripts/quick_demo.sh`) | no |
| `data/pheme/` | The full corpus (5,802 threads, 27 MB compressed), created by `python -m src.data.download` | no (`.gitignore`) |

## Format of one thread
```
<event>/<rumours|non-rumours>/<thread_id>/
    source-tweet/<thread_id>.json     # the original tweet
    reactions/<tweet_id>.json         # every reply, with in_reply_to_status_id_str
```
Each JSON is a Twitter API v1.1 tweet object. The loader (`src/data/pheme.py`) keeps only the text, the
creation time, the reply-to id and three user fields (followers, friends, verified).

## Pre-processing (done in code, nothing to run by hand)
1. Find every thread folder; label = 1 under `rumours/`, 0 under `non-rumours/`.
2. Skip unreadable files (e.g. macOS `._*` files), sort replies by time, keep at most 300 tweets per thread.
3. Rebuild the reply tree from `in_reply_to_status_id_str`; a reply whose parent is missing is attached to the source tweet.
4. Arrival time = minutes after the source tweet (made non-decreasing).
5. Features: TF-IDF → truncated SVD (64-d), **fitted on training threads only**, plus log followers, log friends and verified (3-d).
6. Parsed threads are cached as `_cache_threads_*.pkl` next to the data, so later runs start fast.

To rebuild the sample from the full corpus:
`python -m src.data.make_samples --src data/pheme --per-class 10`
