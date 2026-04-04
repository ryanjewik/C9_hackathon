"""Draw ROI boxes on a specific frame from a VOD for visual debugging."""
import cv2
import sys

# ROI_CONFIG from settings.py — normal (no TEAM COMMS)
ROI_CONFIG = {
    "map_indicator": (0.0, 0.0, 0.32, 0.025),
    "minimap": (0.016, 0.032, 0.250, 0.385),
    "top_hud": (0.335, 0.005, 0.330, 0.200),
    "top_left_score": (0.417, 0.009, 0.036, 0.055),
    "top_center_timer": (0.465, 0.010, 0.070, 0.045),
    "top_right_score": (0.555, 0.009, 0.036, 0.055),
    "top_spike_icon": (0.485, 0.065, 0.035, 0.058),
    "top_plant_text": (0.43, 0.127, 0.14, 0.070),
    "top_left_team_tag": (0.379, 0.007, 0.038, 0.043),
    "top_right_team_tag": (0.587, 0.007, 0.040, 0.043),
    "killfeed": (0.690, 0.092, 0.305, 0.318),
    "team_comms": (0.815, 0.055, 0.150, 0.060),
    "bottom_hud": (0.215, 0.870, 0.570, 0.125),
    "replay_indicator": (0.780, 0.850, 0.210, 0.120),
    "score_bar": (0.350, 0.010, 0.300, 0.055),
    "left_panels": (0.000, 0.500, 0.185, 0.500),
    "right_panels": (0.815, 0.500, 0.185, 0.500),
    # Left player cards
    "left_player_1": (0.005, 0.505, 0.175, 0.09),
    "left_player_2": (0.005, 0.605, 0.175, 0.09),
    "left_player_3": (0.005, 0.705, 0.175, 0.09),
    "left_player_4": (0.005, 0.805, 0.175, 0.09),
    "left_player_5": (0.005, 0.905, 0.175, 0.09),
    # Right player cards
    "right_player_1": (0.820, 0.505, 0.175, 0.09),
    "right_player_2": (0.820, 0.605, 0.175, 0.09),
    "right_player_3": (0.820, 0.705, 0.175, 0.09),
    "right_player_4": (0.820, 0.805, 0.175, 0.09),
    "right_player_5": (0.820, 0.905, 0.175, 0.09),
    # Killfeed rows (9 rows)
    "killfeed_row_1": (0.6900, 0.0840, 0.3050, 0.0340),
    "killfeed_row_2": (0.6900, 0.1180, 0.3050, 0.0340),
    "killfeed_row_3": (0.6900, 0.1520, 0.3050, 0.0340),
    "killfeed_row_4": (0.6900, 0.1860, 0.3050, 0.0340),
    "killfeed_row_5": (0.6900, 0.2200, 0.3050, 0.0340),
    "killfeed_row_6": (0.6900, 0.2540, 0.3050, 0.0340),
    "killfeed_row_7": (0.6900, 0.2880, 0.3050, 0.0340),
    "killfeed_row_8": (0.6900, 0.3220, 0.3050, 0.0340),
    "killfeed_row_9": (0.6900, 0.3560, 0.3050, 0.0340),
}

# TEAM COMMS override ROIs
# Game viewport compresses to ~left 56% of screen when TEAM COMMS is active.
# Initial estimates — tune visually with --mode tc.
TEAM_COMMS_OVERRIDES = {
    # Top HUD
    "map_indicator":      (0.000, 0.000, 0.240, 0.025),
    "minimap":            (0.009, 0.032, 0.170, 0.335),
    "top_hud":            (0.280, 0.005, 0.200, 0.140),
    "top_left_score":     (0.315, 0.009, 0.025, 0.042),
    "top_center_timer":   (0.360, 0.010, 0.045, 0.035),
    "top_right_score":    (0.420, 0.009, 0.025, 0.042),
    "top_spike_icon":     (0.365, 0.055, 0.025, 0.045),
    "top_plant_text":     (0.320, 0.090, 0.110, 0.055),
    "score_bar":          (0.290, 0.010, 0.180, 0.045),
    "top_left_team_tag":  (0.285, 0.007, 0.025, 0.035),
    "top_right_team_tag": (0.445, 0.007, 0.025, 0.035),
    # Killfeed
    "killfeed":           (0.520, 0.065, 0.230, 0.250),
    "killfeed_row_1":     (0.520, 0.065, 0.230, 0.028),
    "killfeed_row_2":     (0.520, 0.093, 0.230, 0.028),
    "killfeed_row_3":     (0.520, 0.121, 0.230, 0.028),
    "killfeed_row_4":     (0.520, 0.149, 0.230, 0.028),
    "killfeed_row_5":     (0.520, 0.177, 0.230, 0.028),
    "killfeed_row_6":     (0.520, 0.205, 0.230, 0.028),
    "killfeed_row_7":     (0.520, 0.233, 0.230, 0.028),
    "killfeed_row_8":     (0.520, 0.261, 0.230, 0.028),
    "killfeed_row_9":     (0.520, 0.289, 0.230, 0.028),
    # Left panels & player cards
    "left_panels":        (0.000, 0.385, 0.130, 0.355),
    "left_player_1":      (0.003, 0.385, 0.130, 0.065),
    "left_player_2":      (0.003, 0.457, 0.130, 0.065),
    "left_player_3":      (0.003, 0.531, 0.130, 0.065),
    "left_player_4":      (0.003, 0.607, 0.130, 0.065),
    "left_player_5":      (0.003, 0.680, 0.130, 0.065),
    # Right panels & player cards
    "right_panels":       (0.620, 0.385, 0.130, 0.355),
    "right_player_1":     (0.620, 0.385, 0.130, 0.065),
    "right_player_2":     (0.620, 0.457, 0.130, 0.065),
    "right_player_3":     (0.620, 0.531, 0.130, 0.065),
    "right_player_4":     (0.620, 0.607, 0.130, 0.065),
    "right_player_5":     (0.620, 0.680, 0.130, 0.065),
    # Bottom
    "bottom_hud":         (0.200, 0.670, 0.320, 0.100),
    "replay_indicator":   (0.560, 0.600, 0.170, 0.120),
}

# Color map for different ROI categories
COLORS = {
    "killfeed": (0, 255, 0),       # green
    "team_comms": (0, 255, 255),   # yellow
    "top_hud": (255, 200, 0),      # cyan-ish
    "top_left_score": (255, 200, 0),
    "top_right_score": (255, 200, 0),
    "top_center_timer": (255, 200, 0),
    "top_spike_icon": (255, 200, 0),
    "top_plant_text": (255, 200, 0),
    "top_left_team_tag": (255, 100, 0),
    "top_right_team_tag": (255, 100, 0),
    "score_bar": (255, 200, 0),
    "map_indicator": (200, 200, 200),
    "minimap": (200, 200, 200),
    "bottom_hud": (200, 100, 200),
    "replay_indicator": (0, 0, 255),  # red
    "left_panels": (255, 150, 50),
    "right_panels": (255, 150, 50),
    "left_player_1": (255, 200, 100),
    "left_player_2": (255, 200, 100),
    "left_player_3": (255, 200, 100),
    "left_player_4": (255, 200, 100),
    "left_player_5": (255, 200, 100),
    "right_player_1": (100, 200, 255),
    "right_player_2": (100, 200, 255),
    "right_player_3": (100, 200, 255),
    "right_player_4": (100, 200, 255),
    "right_player_5": (100, 200, 255),
    "killfeed_row_1": (100, 255, 100),
    "killfeed_row_2": (100, 255, 100),
    "killfeed_row_3": (100, 255, 100),
    "killfeed_row_4": (100, 255, 100),
    "killfeed_row_5": (100, 255, 100),
    "killfeed_row_6": (100, 255, 100),
    "killfeed_row_7": (100, 255, 100),
    "killfeed_row_8": (100, 255, 100),
    "killfeed_row_9": (100, 255, 100),
}
DEFAULT_COLOR = (128, 128, 255)


def draw_rois(frame, rois, label_prefix=""):
    h, w = frame.shape[:2]
    for name, (rx, ry, rw, rh) in rois.items():
        px = int(rx * w)
        py = int(ry * h)
        pw = int(rw * w)
        ph = int(rh * h)
        color = COLORS.get(name, DEFAULT_COLOR)
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), color, 2)
        label = f"{label_prefix}{name}"
        # Background for label
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(frame, (px, py - th - 4), (px + tw + 2, py), (0, 0, 0), -1)
        cv2.putText(frame, label, (px + 1, py - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return frame


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vod", default="/app/uploads/match_vod_9.mp4")
    parser.add_argument("--time", default="07:44", help="MM:SS or seconds")
    parser.add_argument("--out", default="/app/outputs/debug_roi_frame_vod9.png")
    parser.add_argument("--mode", default="both", choices=["normal", "tc", "both"],
                        help="normal=standard ROIs, tc=TEAM COMMS adjusted, both=overlay both")
    args = parser.parse_args()

    vod_path = args.vod
    if ":" in args.time:
        parts = args.time.split(":")
        timestamp_s = int(parts[0]) * 60 + int(parts[1])
    else:
        timestamp_s = int(args.time)
    output_path = args.out

    cap = cv2.VideoCapture(vod_path)
    if not cap.isOpened():
        # Try docker path
        print(f"Cannot open {vod_path}, trying alternate paths...")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame = int(timestamp_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Failed to read frame at {timestamp_s}s (frame {target_frame})")
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}, FPS: {fps}, target frame: {target_frame}")

    annotated = frame.copy()

    if args.mode in ("normal", "both"):
        draw_rois(annotated, ROI_CONFIG)

    if args.mode in ("tc", "both"):
        # Build the effective TEAM COMMS ROI set: start with normal, apply overrides
        tc_rois = dict(ROI_CONFIG)
        tc_rois.update(TEAM_COMMS_OVERRIDES)
        # Draw with dashed-style (thinner + different shade) to distinguish
        for name, (rx, ry, rw, rh) in TEAM_COMMS_OVERRIDES.items():
            px = int(rx * w)
            py = int(ry * h)
            pw = int(rw * w)
            ph = int(rh * h)
            # Magenta dashed for TEAM COMMS adjusted
            cv2.rectangle(annotated, (px, py), (px + pw, py + ph), (255, 0, 255), 2)
            label = f"TC:{name}"
            (tw, th2), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(annotated, (px, py + ph), (px + tw + 2, py + ph + th2 + 4), (0, 0, 0), -1)
            cv2.putText(annotated, label, (px + 1, py + ph + th2 + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, annotated)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
