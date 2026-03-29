import re

v25f_detects = []
for vod in range(1, 10):
    path = rf'E:\cloud9_hackathon\extract_log_vod{vod}.txt'
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    for m in re.finditer(r'\[CROP-DBG\] crop#(\d+) ULT BADGE detected \(killer=([0-9.]+)% victim=([0-9.]+)%\)', text):
        v25f_detects.append((vod, int(m.group(1)), float(m.group(2)), float(m.group(3))))

v25g_detects = []
for logfile, enc in [
    (r'E:\cloud9_hackathon\extract_log_all_v25g.txt', 'utf-16'),
    (r'E:\cloud9_hackathon\extract_log_v25g_vod6-9.txt', 'utf-16'),
]:
    with open(logfile, 'r', encoding=enc, errors='replace') as f:
        text = f.read()
    sections = re.split(r'EXTRACTING CROPS: match_vod(?:_(\d+))?\.mp4', text)
    for i in range(1, len(sections), 2):
        vod_num = int(sections[i]) if sections[i] else 1
        section = sections[i+1] if i+1 < len(sections) else ''
        for m in re.finditer(r'\[CROP-DBG\] crop#(\d+) ULT BADGE detected \(killer=([0-9.]+)% victim=([0-9.]+)%\)', section):
            v25g_detects.append((vod_num, int(m.group(1)), float(m.group(2)), float(m.group(3))))

print('v25f ult badge detections:')
for vod, crop, kp, vp in sorted(v25f_detects):
    tag = "  ** killer_pct >= 58% **" if kp >= 58.0 else ""
    print(f'  VOD{vod} crop#{crop}: killer={kp}% victim={vp}%{tag}')

print(f'\nv25g ult badge detections:')
for vod, crop, kp, vp in sorted(v25g_detects):
    print(f'  VOD{vod} crop#{crop}: killer={kp}% victim={vp}%')

v25f_set = set((v, c) for v, c, _, _ in v25f_detects)
v25g_set = set((v, c) for v, c, _, _ in v25g_detects)
removed = v25f_set - v25g_set
added = v25g_set - v25f_set
print(f'\nIn v25f but NOT v25g: {sorted(removed)}')
print(f'In v25g but NOT v25f: {sorted(added)}')
