import os, re

# Check v25f local crops
for vod in [6, 7, 8, 9]:
    folder = rf'E:\cloud9_hackathon\local_crops_vod{vod}'
    if not os.path.exists(folder):
        print(f'VOD{vod}: folder not found')
        continue
    # Count ocr_hybrid files
    ocr_dir = os.path.join(folder, 'ocr_hybrid')
    if not os.path.exists(ocr_dir):
        # Maybe no subdirs, just flat
        files = [f for f in os.listdir(folder) if f.endswith('.png')]
        print(f'VOD{vod}: {len(files)} png files (flat)')
        continue
    files = sorted([f for f in os.listdir(ocr_dir) if f.endswith('.png')])
    # Check for duplicate crop numbers
    crop_nums = {}
    for f in files:
        m = re.search(r'crop_(\d+)', f)
        if m:
            num = m.group(1)
            crop_nums.setdefault(num, []).append(f)
    dups = {k: v for k, v in crop_nums.items() if len(v) > 1}
    print(f'VOD{vod}: {len(files)} files, {len(crop_nums)} unique crop#s, {len(dups)} duplicates')
    for num, flist in sorted(dups.items())[:3]:
        for f in flist:
            print(f'  dup crop#{num}: {f}')

print('\n--- v25g VODs 6-9 ---')
ocr_dir = r'E:\cloud9_hackathon\local_crops_v25g_vod6-9\ocr_hybrid'
files = sorted([f for f in os.listdir(ocr_dir) if f.endswith('.png')])
for vod in [6, 7, 8, 9]:
    vod_files = [f for f in files if f.startswith(f'vod{vod}_')]
    crop_nums = {}
    for f in vod_files:
        m = re.search(r'crop_(\d+)', f)
        if m:
            num = m.group(1)
            crop_nums.setdefault(num, []).append(f)
    dups = {k: v for k, v in crop_nums.items() if len(v) > 1}
    print(f'VOD{vod}: {len(vod_files)} files, {len(crop_nums)} unique crop#s, {len(dups)} duplicates')
    for num, flist in sorted(dups.items())[:3]:
        for f in flist:
            print(f'  dup crop#{num}: {f}')
