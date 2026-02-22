import json, os, sys
from collections import Counter

meta = json.load(open(r"e:\cloud9_hackathon\vod_processor\outputs\crops\crops_meta.json"))
icon_dir = r"e:\cloud9_hackathon\vod_processor\outputs\crops\icons"
import cv2
ws = []
for e in meta:
    p = os.path.join(icon_dir, os.path.basename(e["icon_path"]))
    img = cv2.imread(p)
    if img is not None:
        ws.append(img.shape[1])

sys.stdout.write(f"Total: {len(ws)}\n")
sys.stdout.write(f"Min: {min(ws)}, Max: {max(ws)}, Mean: {sum(ws)/len(ws):.1f}\n")
c = Counter(ws)
items = sorted(c.items())
sys.stdout.write(f"Unique widths: {len(items)}\n")
for w_val, n in items:
    sys.stdout.write(f"  w={w_val}px count={n}\n")
sys.stdout.flush()
