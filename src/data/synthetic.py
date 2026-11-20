"""SMOKE TEST ONLY: writes a tiny fake dataset in the PHEME-RnR folder layout to check the code runs.
Never used for any reported result.   python -m src.data.synthetic data/synth"""
import json, os, random, sys
from datetime import datetime, timedelta, timezone

out = sys.argv[1] if len(sys.argv) > 1 else "data/synth"
rng = random.Random(0); FMT = "%a %b %d %H:%M:%S %z %Y"; tid = 1000
W = {1: "unconfirmed claims reports allegedly shooting hostage breaking is this true source ? really fake".split(),
     0: "police confirmed statement official update thoughts prayers news report sad terrible".split()}
for event in ["eventA", "eventB", "eventC"]:
    for lab, name in [(1, "rumours"), (0, "non-rumours")]:
        for _ in range(60 if lab else 100):
            tid += 1; root = os.path.join(out, "pheme-rnr-dataset", event, name, str(tid))
            os.makedirs(os.path.join(root, "source-tweet")); os.makedirs(os.path.join(root, "reactions"))
            t0 = datetime(2015, 1, 7, 11, 0, tzinfo=timezone.utc); ids = [tid]
            def tw(i, parent, t):
                txt = " ".join(rng.choice(W[lab] if rng.random() < 0.6 else W[1 - lab]) for _ in range(8))
                return dict(id_str=str(i), text=txt, created_at=t.strftime(FMT), in_reply_to_status_id_str=parent,
                            user=dict(followers_count=rng.randint(1, 5000), friends_count=rng.randint(1, 900), verified=False))
            json.dump(tw(tid, None, t0), open(os.path.join(root, "source-tweet", f"{tid}.json"), "w"))
            t = t0
            for _ in range(rng.randint(3, 30 if lab else 18)):
                tid += 1; t += timedelta(minutes=rng.expovariate(1 / (4 if lab else 9)))
                json.dump(tw(tid, str(rng.choice(ids)), t), open(os.path.join(root, "reactions", f"{tid}.json"), "w")); ids.append(tid)
            open(os.path.join(root, "reactions", "._junk.json"), "w").write("not json")
print("synthetic smoke-test data ->", out)
