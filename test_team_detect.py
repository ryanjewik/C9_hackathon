"""Quick test: only run team tag auto-detection on VODs 4-7.
Does NOT process the full video — just detects teams and exits.
"""
import os, sys, cv2
sys.path.insert(0, '/app')

from app.services.processing.vod_processor import VODProcessor

EXPECTED = {
    4: ("FNC", "PRX"),
    5: ("BLG", "WOL"),
    6: ("TL",  "SEN"),
    7: ("SEN", "EG"),
}

for vod in [4, 5, 6, 7]:
    if vod == 1:
        path = "/app/uploads/match_vod.mp4"
    else:
        path = f"/app/uploads/match_vod_{vod}.mp4"

    if not os.path.exists(path):
        print(f"VOD {vod}: SKIP (not found)")
        continue

    print(f"\n{'='*60}")
    print(f"VOD {vod}  —  expected: {EXPECTED.get(vod, '?')}")
    print(f"{'='*60}")

    proc = VODProcessor()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize player matcher so validation can update it
    from app.services.db.db_player_matcher import DatabasePlayerMatcher
    proc._player_matcher = DatabasePlayerMatcher()

    # Step 1: OCR-based tag detection
    left, right, left_cands, right_cands = proc._detect_team_tags_from_hud(cap, fps)
    print(f"  OCR tags -> left={left!r}  right={right!r}")

    # Apply the detected tags so validation can check them
    proc._left_team_code = left
    proc._right_team_code = right
    if hasattr(proc, '_player_matcher') and proc._player_matcher:
        proc._player_matcher._left_team_code = left
        proc._player_matcher._right_team_code = right

    # Load player pools for the OCR-detected teams
    from vod_processor.app.services.db.db_player_matcher import load_match_players_from_db
    try:
        lp, rp = load_match_players_from_db(left or "", right or "")
        proc._left_player_pool = lp or []
        proc._right_player_pool = rp or []
    except:
        proc._left_player_pool = []
        proc._right_player_pool = []

    # Step 2: Validation + Phase 2 fallback
    job_id = f"test-vod{vod}"
    proc._validate_team_via_players(job_id, cap, fps, left_cands, right_cands)

    final_left = proc._left_team_code
    final_right = proc._right_team_code
    exp = EXPECTED.get(vod, (None, None))

    left_ok = "OK" if final_left == exp[0] else "WRONG"
    right_ok = "OK" if final_right == exp[1] else "WRONG"

    print(f"\n  FINAL: left={final_left} [{left_ok}]  right={final_right} [{right_ok}]")
    cap.release()

print("\nDone.")
