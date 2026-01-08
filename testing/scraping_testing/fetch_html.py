import os
import time
import requests
from urllib.parse import urlparse, parse_qs

OUTDIR = "HTML_samples"
os.makedirs(OUTDIR, exist_ok=True)

URLS = [
    ("https://www.vlr.gg/events/?tier=60", "events_tier_60.html"),
    ("https://www.vlr.gg/event/2283/valorant-champions-2025", "event_2283_valorant_champions_2025.html"),
    ("https://www.vlr.gg/event/2283/valorant-champions-2025/group-stage", "event_2283_group_stage.html"),
    ("https://www.vlr.gg/event/matches/2283/valorant-champions-2025/?series_id=all", "event_2283_matches_series_all.html"),
    ("https://www.vlr.gg/542195/paper-rex-vs-xi-lai-gaming-valorant-champions-2025-opening-a", "match_542195_paper-rex_vs_xi-lai-gaming_opening_a.html"),
    ("https://www.vlr.gg/team/624/paper-rex", "team_624_paper-rex.html"),
    ("https://www.vlr.gg/player/17086/something", "player_17086_something.html"),
    ("https://www.vlr.gg/542195/paper-rex-vs-xi-lai-gaming-valorant-champions-2025-opening-a/?game=233397&tab=overview", "match_542195_game_233397_overview.html"),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; fetch_html_script/1.0)"
}

DELAY_SEC = 0.3  # safe rate limiting (about 3 requests/sec)


def save_url(url, filename):
    outpath = os.path.join(OUTDIR, filename)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        # Try to decode using apparent encoding, fallback to utf-8
        resp.encoding = resp.apparent_encoding or 'utf-8'
        with open(outpath, 'w', encoding='utf-8') as f:
            f.write(resp.text)
        print(f"Saved: {outpath}")
        return True
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return False


def main():
    print(f"Fetching {len(URLS)} pages -> {OUTDIR} (delay {DELAY_SEC}s)")
    for url, fname in URLS:
        save_url(url, fname)
        time.sleep(DELAY_SEC)


if __name__ == '__main__':
    main()
