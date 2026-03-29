import os, re

for folder, label in [
    (r'E:\cloud9_hackathon\local_crops_v25g', 'v25g VODs 1-5'),
    (r'E:\cloud9_hackathon\local_crops_v25g_vod6-9', 'v25g VODs 6-9'),
]:
    print(f'\n=== {label} ===')
    if not os.path.exists(folder):
        print('  FOLDER NOT FOUND')
        continue
    vod_counts = {}
    method_counts = {}
    for root, dirs, files in os.walk(folder):
        subdir = os.path.relpath(root, folder)
        for f in files:
            if f.endswith('.png'):
                m = re.match(r'vod(\d+)_', f)
                vod = int(m.group(1)) if m else -1
                vod_counts[vod] = vod_counts.get(vod, 0) + 1
                method = subdir.split(os.sep)[0] if subdir != '.' else 'root'
                key = (vod, method)
                method_counts[key] = method_counts.get(key, 0) + 1

    for vod in sorted(vod_counts):
        print(f'  VOD{vod}: {vod_counts[vod]} total files')
        for (v, m), c in sorted(method_counts.items()):
            if v == vod:
                print(f'    {m}: {c}')
