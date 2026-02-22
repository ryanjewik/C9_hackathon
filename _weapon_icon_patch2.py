"""Patch: replace _extract_weapon_icon with wider color ranges for faded killfeed entries."""
TARGET = r'e:\cloud9_hackathon\vod_processor\app\services\processing\vod_processor.py'

NEW_METHOD = '''    def _extract_weapon_icon(self, row_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract the weapon/ability icon from a killfeed row image.

        Killfeed row structure:
          [Agent] [Killer Name on colored bg] [WEAPON ICON] [arrow] [Victim on colored bg] [Agent]

        Strategy:
        1. Build a team-color mask with WIDE HSV ranges (killfeed name
           backgrounds fade as entries age, lowering saturation).  Apply
           morphological closing to bridge text-shaped holes.
        2. Compute column-wise color density and threshold to find colored
           "runs" (name-background bands).
        3. Among all *interior* gaps between consecutive runs, pick the
           largest one -- that is the weapon/ability icon region.
        4. Keep full row height, add small horizontal padding, crop.

        Position varies with player name length and icon size, so we avoid
        any fixed-offset heuristics.
        """
        try:
            h, w = row_img.shape[:2]
            if w < 60 or h < 10:
                return None

            # -- Step 1: Build combined team-color mask --
            # Use WIDER ranges than the global TEAM_COLORS because killfeed
            # name backgrounds fade (lower saturation/value) as entries age.
            hsv = cv2.cvtColor(row_img, cv2.COLOR_BGR2HSV)

            # Teal/cyan: widen saturation down to 25 (was 50) and value to 60 (was 80)
            teal_mask = cv2.inRange(
                hsv, np.array([75, 25, 60]), np.array([120, 255, 255])
            )
            # Orange: widen sat to 60 (was 80) and hue to 30 (was 25)
            orange_mask1 = cv2.inRange(
                hsv, np.array([0, 60, 80]), np.array([30, 255, 255])
            )
            # Orange wrap-around
            orange_mask2 = cv2.inRange(
                hsv, np.array([155, 60, 80]), np.array([180, 255, 255])
            )
            color_mask = cv2.bitwise_or(
                teal_mask, cv2.bitwise_or(orange_mask1, orange_mask2)
            )

            # -- Step 1b: Morphological closing to bridge text-shaped gaps --
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

            # -- Step 5: Horizontal padding only (keep full row height) --
            pad_x = max(2, int(best_gap_w * 0.08))
            x0 = max(0, gap_left - pad_x)
            x1 = min(w, gap_right + pad_x)

            icon = row_img[0:h, x0:x1]
            if icon.size == 0:
                return None
            return icon

        except Exception:
            return None

'''

with open(TARGET, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find method boundaries
start = None
end = None
for i, line in enumerate(lines):
    if 'def _extract_weapon_icon(self, row_img' in line:
        start = i
    if start is not None and i > start and 'def _center_fallback_crop' in line:
        end = i
        break

if start is None or end is None:
    raise RuntimeError(f"Could not find method boundaries: start={start}, end={end}")

new_lines = [l + '\\n' for l in NEW_METHOD.split('\\n')]

patched = lines[:start] + new_lines + lines[end:]
with open(TARGET, 'w', encoding='utf-8') as f:
    f.writelines(patched)

print(f"Patched lines {start+1}-{end} with {len(new_lines)} new lines")
print(f"Old: {len(lines)} lines, New: {len(patched)} lines")
