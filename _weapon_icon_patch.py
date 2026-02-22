"""Patch script: replace _extract_weapon_icon in vod_processor.py"""
import sys

TARGET = r'e:\cloud9_hackathon\vod_processor\app\services\processing\vod_processor.py'

NEW_METHOD = r'''    def _extract_weapon_icon(self, row_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract the weapon/ability icon from a killfeed row image.

        Killfeed row structure:
          [Agent] [Killer Name on colored bg] [WEAPON ICON] [arrow] [Victim on colored bg] [Agent]

        Strategy:
        1. Build a team-color mask and apply morphological closing to bridge
           text-shaped holes (white text on colored bg).
        2. Compute column-wise color density and threshold to find colored
           "runs" (name-background bands).
        3. Among all *interior* gaps between consecutive runs, pick the
           largest one -- that is the weapon/ability icon region.
        4. Refine vertical bounds with edge content, add small padding, crop.

        Position varies with player name length and icon size, so we avoid
        any fixed-offset heuristics.
        """
        try:
            h, w = row_img.shape[:2]
            if w < 60 or h < 10:
                return None

            # -- Step 1: Build combined team-color mask --
            hsv = cv2.cvtColor(row_img, cv2.COLOR_BGR2HSV)
            teal_mask = cv2.inRange(
                hsv,
                np.array(TEAM_COLORS['teal']['lower']),
                np.array(TEAM_COLORS['teal']['upper']),
            )
            orange_mask1 = cv2.inRange(
                hsv,
                np.array(TEAM_COLORS['orange']['lower']),
                np.array(TEAM_COLORS['orange']['upper']),
            )
            orange_mask2 = np.zeros_like(teal_mask)
            if 'lower2' in TEAM_COLORS['orange'] and 'upper2' in TEAM_COLORS['orange']:
                orange_mask2 = cv2.inRange(
                    hsv,
                    np.array(TEAM_COLORS['orange']['lower2']),
                    np.array(TEAM_COLORS['orange']['upper2']),
                )
            color_mask = cv2.bitwise_or(
                teal_mask, cv2.bitwise_or(orange_mask1, orange_mask2)
            )

            # -- Step 1b: Morphological closing to bridge text-shaped gaps --
            # White text on the colored name-bg punches holes in the mask.
            # A wide horizontal kernel fills those holes without bridging
            # the weapon-icon gap (which is typically 40-80+ px wide).
            close_kw = max(3, int(w * 0.04))   # ~23 px at 585 w
            close_kh = max(1, h // 3)           # ~12 px at 38 h
            close_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (close_kw, close_kh)
            )
            color_mask = cv2.morphologyEx(
                color_mask, cv2.MORPH_CLOSE, close_kernel
            )

            # -- Step 2: Column-wise color density (after closing) --
            col_density = np.sum(color_mask > 0, axis=0).astype(np.float64) / max(1, h)

            # Heavier smoothing to consolidate fragments
            ks = max(5, int(w * 0.03) | 1)  # ensure odd, ~17 px
            col_smooth = np.convolve(col_density, np.ones(ks) / ks, mode='same')

            # -- Step 3: Identify "name background" columns --
            DENSITY_THRESH = 0.15
            is_name = col_smooth >= DENSITY_THRESH

            # Collect contiguous runs
            runs: list = []
            in_run = False
            run_start = 0
            for x in range(w):
                if is_name[x] and not in_run:
                    in_run = True
                    run_start = x
                elif not is_name[x] and in_run:
                    in_run = False
                    runs.append((run_start, x))
            if in_run:
                runs.append((run_start, w))

            # Drop tiny noise runs
            MIN_RUN_W = max(8, int(w * 0.02))
            runs = [(s, e) for s, e in runs if (e - s) >= MIN_RUN_W]

            if len(runs) < 2:
                return self._center_fallback_crop(row_img)

            # -- Step 4: Find the largest INTERIOR gap --
            # Gaps at the far edges are agent portraits, not the weapon icon.
            # We want the biggest gap whose centre is in the middle ~70% of
            # the row width.
            best_gap = None
            best_gap_w = 0
            for i in range(len(runs) - 1):
                gap_l = runs[i][1]
                gap_r = runs[i + 1][0]
                gw = gap_r - gap_l
                gap_cx = (gap_l + gap_r) / 2.0
                # Reject gaps whose centre is in the outer 15% on each side
                if gap_cx < w * 0.15 or gap_cx > w * 0.85:
                    continue
                if gw > best_gap_w:
                    best_gap_w = gw
                    best_gap = (gap_l, gap_r)

            if best_gap is None or best_gap_w < 6 or best_gap_w > int(w * 0.55):
                return self._center_fallback_crop(row_img)

            gap_left, gap_right = best_gap

            # -- Step 5: Refine vertical bounds using edge content in gap --
            gap_roi_gray = cv2.cvtColor(
                row_img[:, gap_left:gap_right], cv2.COLOR_BGR2GRAY
            )
            edges_gap = cv2.Canny(gap_roi_gray, 40, 140)
            row_has_edge = np.any(edges_gap > 0, axis=1)
            edge_rows = np.where(row_has_edge)[0]

            if edge_rows.size > 0:
                y0 = max(0, int(edge_rows[0]) - 2)
                y1 = min(h, int(edge_rows[-1]) + 3)
            else:
                pad_y = max(1, int(h * 0.1))
                y0 = pad_y
                y1 = h - pad_y

            # -- Step 6: Small horizontal padding --
            pad_x = max(2, int(best_gap_w * 0.08))
            x0 = max(0, gap_left - pad_x)
            x1 = min(w, gap_right + pad_x)

            icon = row_img[y0:y1, x0:x1]
            if icon.size == 0:
                return None
            return icon

        except Exception:
            return None

'''

with open(TARGET, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Lines 2917-3038 (1-based) are the old method
# Line 2917 = index 2916, up to but NOT including line 3039 (index 3038)
start_idx = 2916  # def _extract_weapon_icon
end_idx = 3038    # line before def _center_fallback_crop

new_lines = NEW_METHOD.split('\n')
# Add newline to each line
new_lines = [l + '\n' for l in new_lines]

# Replace
patched = lines[:start_idx] + new_lines + lines[end_idx:]

with open(TARGET, 'w', encoding='utf-8') as f:
    f.writelines(patched)

print(f"Patched: replaced lines {start_idx+1}-{end_idx} with {len(new_lines)} new lines")
print(f"Old total: {len(lines)}, New total: {len(patched)}")
