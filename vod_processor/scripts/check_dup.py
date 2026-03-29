import json, glob
for f in sorted(glob.glob('/app/outputs/crops-vod7*_events.json'))[-1:]:
    events = json.load(open(f))
    kills = [e for e in events if e['type']=='KILL_EVENT']
    # Find kills near t=2823-2825s
    for k in kills:
        t = k['t_ms']/1000
        if 2820 <= t <= 2830:
            p = k['payload']
            print(f"  t={t:.1f}s {p['killer_name']} -> {p['victim_name']} weapon={p.get('weapon','?')} self={p.get('is_self_kill',False)}")
