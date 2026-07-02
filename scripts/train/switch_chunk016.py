#!/usr/bin/env python3
"""Switch dataset to chunk-016 for smoke test."""
import json, os

META = "/data/droid_mirror/meta"

# Filter episodes to chunk-016 only (index 16000-16999)
filtered = []
with open(f"{META}/episodes.jsonl.full") as f:
    for line in f:
        ep = json.loads(line)
        if 16000 <= ep["episode_index"] < 17000:
            filtered.append(ep)

with open(f"{META}/episodes.jsonl", "w") as f:
    for ep in filtered:
        f.write(json.dumps(ep) + "\n")

# Update info.json
info = json.load(open(f"{META}/info.json.full"))
info["total_episodes"] = 17000  # cover index 0-16999
info["total_chunks"] = 58
info["chunks_size"] = 1000
with open(f"{META}/info.json", "w") as f:
    json.dump(info, f, indent=2)

print(f"Set to chunk-016: {len(filtered)} episodes")
print(f"Index range: {filtered[0]['episode_index']}-{filtered[-1]['episode_index']}")
print(f"total_episodes=17000")
