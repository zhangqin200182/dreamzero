#!/usr/bin/env python3
"""Restore original episode indices and pad trajectory_lengths array."""
import json, os, shutil

META = "/data/droid_mirror/meta"
DATA = "/data/droid_mirror/data"
VIDS = "/data/droid_mirror/videos"

# Restore original episodes from .full backup
full_path = f"{META}/episodes.jsonl.full"
if os.path.exists(full_path):
    shutil.copy(full_path, f"{META}/episodes.jsonl")
    print("Restored episodes.jsonl from .full backup")

full_info = f"{META}/info.json.full"
if os.path.exists(full_info):
    shutil.copy(full_info, f"{META}/info.json")
    print("Restored info.json from .full backup")

# Only keep chunk-001 episodes (index 1000-1999)
filtered = []
with open(f"{META}/episodes.jsonl") as f:
    for line in f:
        ep = json.loads(line)
        if 1000 <= ep["episode_index"] < 2000:
            filtered.append(ep)

with open(f"{META}/episodes.jsonl", "w") as f:
    for ep in filtered:
        f.write(json.dumps(ep) + "\n")

print(f"Filtered to {len(filtered)} episodes, range {filtered[0]['episode_index']}-{filtered[-1]['episode_index']}")

# Move parquets back from chunk-000 to chunk-001 with original names
chunk0 = f"{DATA}/chunk-000"
chunk1 = f"{DATA}/chunk-001"
os.makedirs(chunk1, exist_ok=True)

if os.path.exists(chunk0):
    # We renamed 1000-1999 to 0-999 earlier, need to reverse
    for f_name in os.listdir(chunk0):
        if f_name.endswith(".parquet"):
            new_idx = int(f_name.replace("episode_", "").replace(".parquet", ""))
            old_idx = new_idx + 1000  # reverse the shift
            src = os.path.join(chunk0, f_name)
            dst = os.path.join(chunk1, f"episode_{old_idx:06d}.parquet")
            shutil.move(src, dst)

print(f"Parquets in chunk-001: {len(os.listdir(chunk1))}")

# Videos: move chunk-000 back to chunk-001
chunk0_v = f"{VIDS}/chunk-000"
chunk1_v = f"{VIDS}/chunk-001"

if os.path.exists(chunk0_v):
    real_view = "observation.images.exterior_image_2_left"
    src_view = os.path.join(chunk0_v, real_view)
    dst_view = os.path.join(chunk1_v, real_view)
    os.makedirs(dst_view, exist_ok=True)

    if os.path.isdir(src_view) and not os.path.islink(src_view):
        for vf in os.listdir(src_view):
            if vf.endswith(".mp4"):
                new_idx = int(vf.replace("episode_", "").replace(".mp4", ""))
                old_idx = new_idx + 1000
                src = os.path.join(src_view, vf)
                dst = os.path.join(dst_view, f"episode_{old_idx:06d}.mp4")
                if os.path.islink(src):
                    target = os.readlink(src)
                    t_idx = int(target.replace("episode_", "").replace(".mp4", ""))
                    os.symlink(f"episode_{t_idx + 1000:06d}.mp4", dst)
                    os.remove(src)
                else:
                    shutil.move(src, dst)

    # Clean up chunk-000
    shutil.rmtree(chunk0_v, ignore_errors=True)

# Ensure symlinks for other views in chunk-001
for view in ["observation.images.exterior_image_1_left", "observation.images.wrist_image_left"]:
    link = os.path.join(chunk1_v, view)
    if os.path.islink(link):
        os.remove(link)
    if not os.path.exists(link):
        os.symlink(real_view, link)

print(f"Videos in chunk-001/{real_view}: {len([f for f in os.listdir(dst_view) if f.endswith('.mp4')])}")

# Update info.json: set total_episodes = 2000 so trajectory_lengths[1999] works
info = json.load(open(f"{META}/info.json"))
info["total_episodes"] = 2000  # Must cover index 0-1999
info["total_chunks"] = 58
info["chunks_size"] = 1000
with open(f"{META}/info.json", "w") as f:
    json.dump(info, f, indent=2)

print(f"Set total_episodes=2000 to cover index range 0-1999")
print("Done")
