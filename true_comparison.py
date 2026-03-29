import os, re

# v25g VODs 1-5 
ocr_dir = r'E:\cloud9_hackathon\local_crops_v25g\ocr_hybrid'
if os.path.exists(ocr_dir):
    files = sorted([f for f in os.listdir(ocr_dir) if f.endswith('.png')])
    print(f'--- v25g VODs 1-5 ocr_hybrid ---')
    for vod in [1, 2, 3, 4, 5]:
        vod_files = [f for f in files if f.startswith(f'vod{vod}_')]
        crop_nums = {}
        for f in vod_files:
            m = re.search(r'crop_(\d+)', f)
            if m:
                num = m.group(1)
                crop_nums.setdefault(num, []).append(f)
        dups = {k: v for k, v in crop_nums.items() if len(v) > 1}
        print(f'  VOD{vod}: {len(vod_files)} files, {len(crop_nums)} unique crop#s, {len(dups)} duplicates')
        for num, flist in sorted(dups.items())[:2]:
            for f in flist:
                print(f'    dup crop#{num}: {f}')

# Now compare: v25f vs v25g TRUE unique counts (all VODs)
print('\n=== TRUE COMPARISON (unique crop numbers) ===')
v25f_unique = {}
for vod in range(1, 10):
    folder = rf'E:\cloud9_hackathon\local_crops_vod{vod}'
    if not os.path.exists(folder):
        v25f_unique[vod] = 0
        continue
    count = 0
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith('.png') and ('crop_' in f or f.startswith('crop_')):
                # Only count non-diag
                if 'row_' not in f:
                    count += 1
    v25f_unique[vod] = count

v25g_unique = {}
# VODs 1-5
ocr_dir = r'E:\cloud9_hackathon\local_crops_v25g\ocr_hybrid'
if os.path.exists(ocr_dir):
    files = [f for f in os.listdir(ocr_dir) if f.endswith('.png')]
    for vod in [1, 2, 3, 4, 5]:
        vod_files = [f for f in files if f.startswith(f'vod{vod}_')]
        nums = set()
        for f in vod_files:
            m = re.search(r'crop_(\d+)', f)
            if m:
                nums.add(m.group(1))
        v25g_unique[vod] = len(nums)

# VODs 6-9
ocr_dir = r'E:\cloud9_hackathon\local_crops_v25g_vod6-9\ocr_hybrid'
if os.path.exists(ocr_dir):
    files = [f for f in os.listdir(ocr_dir) if f.endswith('.png')]
    for vod in [6, 7, 8, 9]:
        vod_files = [f for f in files if f.startswith(f'vod{vod}_')]
        nums = set()
        for f in vod_files:
            m = re.search(r'crop_(\d+)', f)
            if m:
                nums.add(m.group(1))
        v25g_unique[vod] = len(nums)

print(f'{"VOD":>4} {"v25f":>6} {"v25g":>6} {"delta":>6}')
total_f, total_g = 0, 0
for vod in range(1, 10):
    f_count = v25f_unique.get(vod, 0)
    g_count = v25g_unique.get(vod, 0)
    delta = g_count - f_count
    total_f += f_count
    total_g += g_count
    print(f'{vod:>4} {f_count:>6} {g_count:>6} {delta:>+6}')
print(f'{"TOT":>4} {total_f:>6} {total_g:>6} {total_g-total_f:>+6}')
