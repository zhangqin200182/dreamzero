#!/usr/bin/env python3
"""Fast parallel download of DreamZero-DROID-Data using hf-mirror.com.

Uses the local /data/droid directory (with LFS pointers) to get the file list,
then downloads actual data via hf_hub_download from the Chinese mirror.
"""
import os
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import hf_hub_download

REPO_ID = "GEAR-Dreams/DreamZero-DROID-Data"
SRC_DIR = "/data/droid"
DST_DIR = "/data/droid_mirror"
MAX_WORKERS = int(os.environ.get("DL_WORKERS", "16"))

parquet_only = "--parquet-only" in sys.argv

print(f"Scanning {SRC_DIR} for file list...", flush=True)
t0 = time.time()
all_files = []
for root, dirs, files in os.walk(SRC_DIR):
    for fname in files:
        full = os.path.join(root, fname)
        rel = os.path.relpath(full, SRC_DIR)
        if parquet_only and not rel.endswith(".parquet"):
            continue
        all_files.append(rel)

print(f"Found {len(all_files)} files in {time.time()-t0:.1f}s", flush=True)

to_download = []
for f in all_files:
    target = os.path.join(DST_DIR, f)
    if os.path.exists(target) and os.path.getsize(target) > 200:
        continue
    to_download.append(f)

print(f"To download: {len(to_download)} (skipping {len(all_files)-len(to_download)} already done)", flush=True)

if not to_download:
    print("DOWNLOAD_COMPLETE: nothing to download", flush=True)
    sys.exit(0)

downloaded = 0
errors = 0
total_bytes = 0
lock = threading.Lock()

def download_one(filename):
    global downloaded, errors, total_bytes
    for attempt in range(5):
        try:
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=DST_DIR,
            )
            sz = os.path.getsize(path) if path and os.path.exists(path) else 0
            with lock:
                downloaded += 1
                total_bytes += sz
            return True
        except Exception as e:
            if attempt == 4:
                with lock:
                    errors += 1
                print(f"FAILED: {filename}: {e}", flush=True)
                return False
            time.sleep(3 * (2 ** attempt))

start = time.time()
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futures = {pool.submit(download_one, f): f for f in to_download}
    for i, future in enumerate(as_completed(futures)):
        future.result()
        if (i + 1) % 200 == 0 or (i + 1) == len(to_download):
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (len(to_download) - i - 1) / rate if rate > 0 else 0
            mb = total_bytes / 1e6
            mbps = mb / elapsed if elapsed > 0 else 0
            print(
                f"Progress: {i+1}/{len(to_download)} "
                f"({downloaded} ok, {errors} err) "
                f"{rate:.1f} files/s  {mb:.0f} MB ({mbps:.1f} MB/s)  "
                f"ETA: {remaining/60:.0f} min",
                flush=True,
            )

elapsed = time.time() - start
print(f"DOWNLOAD_COMPLETE: {downloaded} ok, {errors} errors, {total_bytes/1e9:.1f} GB, {elapsed:.0f}s total", flush=True)
