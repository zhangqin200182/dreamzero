#!/usr/bin/env python3
"""Re-index DreamZero dataset to use episode indices starting from 0."""
import json, os, shutil

META = "/data/droid_mirror/meta"
DATA = "/data/droid_mirror/data"
VIDS = "/data/droid_mirror/videos"

# Read current episodes
episodes = []
with open(f"{META}/episodes.jsonl") as f:
    for line in f:
        episodes.append(json.loads(line))

print(f"Original: {len(episodes)} episodes, range {episodes[0]['episode_index']}-{episodes[-1]['episode_index']}")

# Build old->new mapping
old_to_new = {}
for i, ep in enumerate(episodes):
    old_to_new[ep["episode_index"]] = i
    ep["episode_index"] = i

# Write re-indexed episodes
with open(f"{META}/episodes.jsonl", "w") as f:
    for ep in episodes:
        f.write(json.dumps(ep) + "\n")

# Move parquets: chunk-001 -> chunk-000 with new names
src_chunk = f"{DATA}/chunk-001"
dst_chunk = f"{DATA}/chunk-000"
os.makedirs(dst_chunk, exist_ok=True)

for old_idx, new_idx in old_to_new.items():
    src = f"{src_chunk}/episode_{old_idx:06d}.parquet"
    dst = f"{dst_chunk}/episode_{new_idx:06d}.parquet"
    if os.path.exists(src):
        shutil.move(src, dst)

print(f"Parquets in chunk-000: {len(os.listdir(dst_chunk))}")

# Handle videos: only the real directory (exterior_image_2_left)
real_view = "observation.images.exterior_image_2_left"
src_vdir = f"{VIDS}/chunk-001/{real_view}"
dst_vdir = f"{VIDS}/chunk-000/{real_view}"
os.makedirs(dst_vdir, exist_ok=True)

for vf in os.listdir(src_vdir):
    if vf.endswith(".mp4"):
        old_idx = int(vf.replace("episode_", "").replace(".mp4", ""))
        if old_idx in old_to_new:
            new_idx = old_to_new[old_idx]
            src = f"{src_vdir}/{vf}"
            dst = f"{dst_vdir}/episode_{new_idx:06d}.mp4"
            if os.path.islink(src):
                target = os.readlink(src)
                t_idx = int(target.replace("episode_", "").replace(".mp4", ""))
                if t_idx in old_to_new:
                    os.symlink(f"episode_{old_to_new[t_idx]:06d}.mp4", dst)
                else:
                    os.symlink(target, dst)
            else:
                shutil.move(src, dst)

# Create symlinks for other views
for view in ["observation.images.exterior_image_1_left", "observation.images.wrist_image_left"]:
    link = f"{VIDS}/chunk-000/{view}"
    if not os.path.exists(link):
        os.symlink(real_view, link)

print(f"Videos in chunk-000/{real_view}: {len(os.listdir(dst_vdir))}")
print(f"Views: {os.listdir(f'{VIDS}/chunk-000/')}")

# Update info.json
info = json.load(open(f"{META}/info.json"))
info["total_episodes"] = len(episodes)
info["total_chunks"] = 1
info["chunks_size"] = 1000
with open(f"{META}/info.json", "w") as f:
    json.dump(info, f, indent=2)

# Copy relative_stats_dreamzero.json to root if not there
root_stats = "/data/droid_mirror/relative_stats_dreamzero.json"
if not os.path.exists(root_stats):
    shutil.copy(f"{META}/relative_stats_dreamzero.json", root_stats)

print(f"Done: {len(episodes)} episodes, indices 0-{len(episodes)-1}, chunk-000")
