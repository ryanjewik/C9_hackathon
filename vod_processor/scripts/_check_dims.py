import cv2, os, json

d = r'e:\cloud9_hackathon\vod_processor\outputs\crops\icons'
meta = json.loads(open(r'e:\cloud9_hackathon\vod_processor\outputs\crops\crops_meta.json').read())

files = sorted(os.listdir(d))
widths = []
for f in files:
    img = cv2.imread(os.path.join(d, f))
    if img is not None:
        h, w = img.shape[:2]
        widths.append(w)

# Distribution summary
from collections import Counter
wc = Counter(widths)
print("Icon crop width distribution:")
for w, c in sorted(wc.items()):
    print(f"  {w}px wide: {c} crops")

print(f"\nTotal: {len(widths)} icons")
print(f"Time range: {meta[0]['t_ms']/1000:.0f}s - {meta[-1]['t_ms']/1000:.0f}s")

# Show first 20 files with dims
print("\nFirst 20 icon dims:")
for f in files[:20]:
    img = cv2.imread(os.path.join(d, f))
    if img is not None:
        h, w = img.shape[:2]
        print(f"  {f}: {w}x{h}")
