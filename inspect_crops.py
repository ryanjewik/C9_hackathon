import os, re

folder = r'E:\cloud9_hackathon\local_crops_v25g_vod6-9\ocr_hybrid'
files = sorted(os.listdir(folder))
print(f"Total files: {len(files)}")

# Group by vod prefix  
vod_groups = {}
for f in files:
    m = re.match(r'(vod\d+)_', f)
    prefix = m.group(1) if m else 'no_prefix'
    vod_groups.setdefault(prefix, []).append(f)

for prefix in sorted(vod_groups):
    flist = vod_groups[prefix]
    print(f"\n{prefix}: {len(flist)} files")
    # Show first 5 and last 5
    for f in flist[:3]:
        print(f"  {f}")
    if len(flist) > 6:
        print(f"  ...")
    for f in flist[-3:]:
        print(f"  {f}")
