import cv2, os
d = r'E:\cloud9_hackathon\crops'
files = sorted(os.listdir(d))
widths = []
for f in files:
    img = cv2.imread(os.path.join(d, f))
    if img is not None:
        widths.append((f, img.shape[1], img.shape[0]))

print(f'Total: {len(widths)} crops')
ws = [w for _, w, _ in widths]
print(f'Width range: {min(ws)}-{max(ws)}')
print(f'Avg width: {sum(ws)/len(ws):.0f}')
print()
buckets = {'40-60': 0, '60-80': 0, '80-120': 0, '120-150': 0, '150+': 0}
for _, w, _ in widths:
    if w < 60: buckets['40-60'] += 1
    elif w < 80: buckets['60-80'] += 1
    elif w < 120: buckets['80-120'] += 1
    elif w < 150: buckets['120-150'] += 1
    else: buckets['150+'] += 1
for k, v in buckets.items():
    pct = v * 100 // len(widths)
    print(f'  {k}px: {v} ({pct}%)')
print(f'\nOver 140px: {sum(1 for _, w, _ in widths if w >= 140)}')
print(f'\nFirst 20:')
for f, w, h in widths[:20]:
    print(f'  {f}: {w}x{h}')
