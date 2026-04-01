import re

with open('extract_log_vod8_v3.txt', 'r', encoding='utf-16') as f:
    lines = f.readlines()

# Find kills where the killer was matched via fuzzy at score <= 0.62
for i, line in enumerate(lines):
    line_s = line.strip()
    m = re.search(r"Strict fuzzy: '(.+?)' -> '(.+?)' \(score=([\d.]+)\)", line_s)
    if m and float(m.group(3)) <= 0.62:
        ocr = m.group(1)
        matched = m.group(2)
        score = m.group(3)
        # Check if next few lines have a [KILL] using this matched name as killer
        for j in range(i+1, min(i+8, len(lines))):
            if '[KILL]' in lines[j] and matched in lines[j]:
                kill_line = lines[j].strip()
                # Check if matched name is the killer (before "killed")
                km = re.match(r'\[KILL\] .+: (.+?) killed (.+)', kill_line)
                if km and km.group(1) == matched:
                    print(f'{kill_line}')
                    print(f'  <- killer OCR: "{ocr}" (score={score})')
                    print()
                break
