import json, glob
for f in sorted(glob.glob('/app/outputs/crops-vod7*_events.json')):
    events = json.load(open(f))
    kills = [e for e in events if e['type']=='KILL_EVENT']
    self_kills = [k for k in kills if k['payload'].get('is_self_kill')]
    yay_kills = [k for k in kills if 'yay' in (k['payload'].get('killer_name','') + k['payload'].get('victim_name',''))]
    print(f'File: {f}')
    print(f'Total kills: {len(kills)}, yay-involved: {len(yay_kills)}, self-kills: {len(self_kills)}')
    for sk in self_kills:
        p = sk['payload']
        t = sk['t_ms']/1000
        print(f'  SELF-KILL t={t:.1f}s {p["killer_name"]} -> {p["victim_name"]}')
    for yk in yay_kills:
        p = yk['payload']
        t = yk['t_ms']/1000
        print(f'  yay event t={t:.1f}s {p["killer_name"]} -> {p["victim_name"]} self={p.get("is_self_kill",False)}')
