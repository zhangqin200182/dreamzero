#!/usr/bin/env python3
"""Download all video files for DreamZero-DROID-Data from hf-mirror.com."""
import os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import hf_hub_download

REPO_ID = "GEAR-Dreams/DreamZero-DROID-Data"
SRC_DIR = "/data/droid"
DST_DIR = "/data/droid_mirror"
MAX_WORKERS = int(os.environ.get("DL_WORKERS", "16"))

print("Scanning for video files...", flush=True)
t0 = time.time()
files = []
vdir = os.path.join(SRC_DIR, "videos")
for root, dirs, fnames in os.walk(vdir):
    for f in fnames:
        if f.endswith(".mp4"):
            files.append(os.path.relpath(os.path.join(root, f), SRC_DIR))

print(f"Found {len(files)} video files in {time.time()-t0:.1f}s", flush=True)

to_dl = [f for f in files if not os.path.exists(os.path.join(DST_DIR, f)) or os.path.getsize(os.path.join(DST_DIR, f)) <= 200]
print(f"To download: {len(to_dl)} (skipping {len(files)-len(to_dl)} already done)", flush=True)

if not to_dl:
    print("VIDEO_DOWNLOAD_COMPLETE: nothing to download", flush=True)
    sys.exit(0)

ok = 0; err = 0; tbytes = 0; lock = threading.Lock()
def dl(fn):
    global ok, err, tbytes
    for a in range(5):
        try:
            p = hf_hub_download(repo_id=REPO_ID, filename=fn, repo_type="dataset", local_dir=DST_DIR)
            sz = os.path.getsize(p) if p and os.path.exists(p) else 0
            with lock: ok += 1; tbytes += sz
            return True
        except Exception as e:
            if a == 4:
                with lock: err += 1
                if err <= 20:
                    print(f"FAILED: {fn}: {e}", flush=True)
                return False
            time.sleep(3 * (2 ** a))

t0 = time.time()
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futs = {pool.submit(dl, f): f for f in to_dl}
    for i, fut in enumerate(as_completed(futs)):
        fut.result()
        if (i+1) % 500 == 0 or (i+1) == len(to_dl):
            el = time.time() - t0
            r = (i+1)/el if el > 0 else 0
            rem = (len(to_dl)-i-1)/r/60 if r > 0 else 0
            mb = tbytes/1e6
            print(f"Progress: {i+1}/{len(to_dl)} ({ok} ok, {err} err) {mb:.0f}MB {r:.1f}f/s ETA:{rem:.0f}m", flush=True)

print(f"VIDEO_DOWNLOAD_COMPLETE: {ok} ok, {err} err, {tbytes/1e9:.1f}GB, {time.time()-t0:.0f}s", flush=True)
