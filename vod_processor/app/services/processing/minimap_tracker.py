"""
MinimapTracker — player position tracking on the VALORANT minimap.

Detection pipeline per frame
─────────────────────────────
1. Slide every agent portrait template across the minimap (NCC).
   → Each agent's best-scoring hit becomes a candidate position.
   → Greedy NMS removes duplicates within AGENT_NMS_DIST pixels.
   This step ONLY uses visual agent identity; no colour thresholds,
   so barriers, abilities, etc. cannot produce false player hits.
2. For each hit: sample an annular ring around the centre → teal or
   red pixels determine which team the icon belongs to.
3. Detect spike (always yellow): isolated yellow blob = spike on the
   floor or planted; yellow blob near a player = has_spike flag.
4. Nearest-neighbour track assignment per team; expire missing tracks.
5. Emit POSITION_UPDATE events.

Agent identity strategy
────────────────────────
• AgentTemplateMatcher.find_all() slides each 24×24 portrait template
  across the minimap in one pass and returns the best hit per agent.
• Team assignment uses the coloured ring AROUND each detected icon
  (ring not inside the template, so the two signals are independent).
• Agent name stays on a BlobTrack for the full round once set.
• _agent_to_player persists across rounds: from round 2+, tracks can
  be named on the very first frame if the agent was seen before.
• Kill events are the final source of truth for player↔agent mapping.

Fallback (no agents_dir)
─────────────────────────
• Falls back to HSV colour-blob detection (teal / red rings) when
  agent templates are unavailable.  Agent names will be None.

Spike detection
───────────────
The spike is always bright yellow on the minimap regardless of state
(carried, dropped on floor, or planted).
  • Yellow blob within SPIKE_CARRIER_MAX_DIST of a player → has_spike=True
    on that player's track.
  • Isolated yellow blob → reported as spike_loose in the event payload.

HSV calibration (OpenCV H=0-180, broadcast 1920×1080)
───────────────────────────────────────────────────────
  teal         : H 75-95 , S 50-145, V 80-255
  red (team)   : H 0-15  , S 75-220, V 95-195  +  H 163-180
  spike yellow : H 18-38 , S 150-255, V 180-255
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Optional Pillow for WebP loading fallback
try:
    from PIL import Image as _PIL_Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


# ─────────────────────────────────────────────────────────────────────────────
# Image loader helper
# ─────────────────────────────────────────────────────────────────────────────

def _load_image_bgr(path: str) -> Optional[np.ndarray]:
    """Load an image via OpenCV; fall back to Pillow for WebP if needed."""
    img = cv2.imread(path)
    if img is not None:
        return img
    if _HAS_PIL:
        try:
            pil_img = _PIL_Image.open(path).convert("RGB")
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Agent template matcher
# ─────────────────────────────────────────────────────────────────────────────

class AgentTemplateMatcher:
    """
    Builds 24×24 BGR colour templates from agent portrait files (PNG/WebP)
    and matches minimap icon regions using normalised cross-correlation.

    Using colour (3-channel) NCC is far more discriminative than grayscale
    because Valorant agents have very distinct colour palettes (Raze=orange,
    Sage=teal, Reyna=purple, etc.).  A circular mask sets corner pixels to the
    per-channel inside-circle mean so the map background outside the icon
    contributes zero weight to TM_CCOEFF_NORMED.
    """

    TEMPLATE_SIZE    = 24    # px — template size; higher res than 16 for face detail
    FACE_MASK_RADIUS = 8     # px — only pixels at r ≤ this distance from centre are
                              #      compared.  The team-colour ring sits at r≈9-11 px
                              #      so setting r_mask=8 excludes it entirely, leaving
                              #      only the discriminating face pixels for NCC.
    PORTRAIT_STD_MIN = 18.0  # mean per-channel std-dev of inner face pixels must
                              # exceed this to proceed past portrait guard.
                              # HIGH-quality crops (ncc≥0.85) probe: min=23.9;
                              # background patches: min=11.1 → 6-pt safety margin.
    FACE_HEIGHT_FRAC = 1.0   # fraction of portrait height to use (1.0 = full portrait)
    MIN_CONFIDENCE   = 0.25   # NCC threshold; below → no match
    SEARCH_RADIUS    = 22     # px — search window half-size around blob centroid
                              # must be ≥ RING_SAMPLE_OUTER (20) so arc-centroid
                              # offset from icon centre is always covered

    def __init__(self, agents_dir: str) -> None:
        self._templates: Dict[str, List[np.ndarray]] = {}  # name → [TS×TS×3 uint8 BGR, ...]
        if os.path.isdir(agents_dir):
            self._load_templates(agents_dir)

    @property
    def ready(self) -> bool:
        return len(self._templates) > 0

    def match(self, face_bgr: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Returns (agent_name, score).
        agent_name is None when score < MIN_CONFIDENCE or templates absent.
        """
        if not self.ready or face_bgr is None or face_bgr.size == 0:
            return None, 0.0
        ts = self.TEMPLATE_SIZE
        try:
            resized = cv2.resize(face_bgr, (ts, ts))
        except cv2.error:
            return None, 0.0

        best_name:  Optional[str] = None
        best_score: float         = -1.0
        for name, templs in self._templates.items():
            # Equal-size matchTemplate → 1×1 NCC result in [-1, 1]; max across all crops
            for templ in templs:
                score = float(
                    cv2.matchTemplate(resized, templ, cv2.TM_CCOEFF_NORMED)[0, 0]
                )
                if score > best_score:
                    best_score = score
                    best_name  = name

        if best_score < self.MIN_CONFIDENCE:
            return None, best_score
        return best_name, best_score

    def _load_templates(self, agents_dir: str) -> None:
        ts   = self.TEMPLATE_SIZE
        frac = self.FACE_HEIGHT_FRAC
        r_face = self.FACE_MASK_RADIUS
        # Pre-compute face-only circular mask: r > r_face → ring/background excluded
        _ys, _xs = np.mgrid[0:ts, 0:ts]
        _outside = (_xs - ts // 2) ** 2 + (_ys - ts // 2) ** 2 > r_face ** 2
        self._outside = _outside  # store for patch masking
        _EXTS = {".png", ".jpg", ".jpeg", ".webp"}

        def _make_tmpl(img: np.ndarray) -> np.ndarray:
            h, _ = img.shape[:2]
            face = img[:max(ts, int(h * frac)), :]
            t = cv2.resize(face, (ts, ts)).astype(np.float32)
            for c in range(3):
                ch = t[:, :, c]
                ch[_outside] = float(np.mean(ch[~_outside]))
            return np.clip(t, 0, 255).astype(np.uint8)

        loaded_agents = 0
        total_templates = 0
        for entry in sorted(os.listdir(agents_dir)):
            entry_path = os.path.join(agents_dir, entry)
            if os.path.isdir(entry_path):
                # Skip the spike subfolder — spike templates are handled by
                # SpikeTemplateMatcher, not AgentTemplateMatcher.
                if entry.lower() == "spike":
                    continue
                # Multi-crop mode: subdir named after agent
                name = entry.lower()
                crop_files = [
                    os.path.join(entry_path, f)
                    for f in sorted(os.listdir(entry_path))
                    if os.path.splitext(f)[1].lower() in _EXTS
                ]
            elif os.path.splitext(entry)[1].lower() in _EXTS:
                # Single-file mode: agents_dir/omen.png
                name = os.path.splitext(entry)[0].lower()
                crop_files = [entry_path]
            else:
                continue
            templs = []
            for fpath in crop_files:
                img = _load_image_bgr(fpath)
                if img is not None:
                    templs.append(_make_tmpl(img))
            if templs:
                self._templates[name] = templs
                loaded_agents += 1
                total_templates += len(templs)
        print(f"[AgentTemplateMatcher] Loaded {loaded_agents} agents "
              f"({total_templates} templates) from {agents_dir}")

    # ── Centered-crop matching ────────────────────────────────────────────────
    _NMS_DIST  = 14   # px — cross-agent NMS suppression radius
    _SMALL_SR  = 3    # px — ±search radius; SR=3 compensates for residual arc-centroid
                      #      offset after HoughCircles refinement (was 2).

    def _extract_search_roi(
        self, minimap_bgr: np.ndarray, cx_px: float, cy_px: float
    ) -> np.ndarray:
        """Extract (TS + 2*SR) × (TS + 2*SR) ROI centred on the blob for the
        small sliding-window NCC search.  Uses integer pixel access (no
        subpixel interpolation) because the sliding window itself compensates
        for fractional centroid errors."""
        ts = self.TEMPLATE_SIZE
        sr = self._SMALL_SR
        roi_size = ts + 2 * sr
        half     = roi_size // 2
        cx, cy   = round(cx_px), round(cy_px)
        h, w     = minimap_bgr.shape[:2]
        x1 = max(0, cx - half); y1 = max(0, cy - half)
        x2 = min(w, cx + half); y2 = min(h, cy + half)
        roi = minimap_bgr[y1:y2, x1:x2]
        pad_t = max(0, half - (cy - y1)); pad_b = max(0, half - (y2 - cy))
        pad_l = max(0, half - (cx - x1)); pad_r = max(0, half - (x2 - cx))
        if pad_t or pad_b or pad_l or pad_r:
            roi = cv2.copyMakeBorder(
                roi, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101
            )
        return roi

    def _masked_patch(
        self, minimap_bgr: np.ndarray, cx_px: int, cy_px: int
    ) -> np.ndarray:
        """Extract TS×TS centered on (cx_px, cy_px), then apply the circular
        mask: pixels outside the inscribed circle are set to their per-channel
        mean so they contribute zero to NCC.  This neutralises the team-colour
        ring and map-background corners without altering the face region."""
        ts   = self.TEMPLATE_SIZE
        half = ts // 2
        h, w = minimap_bgr.shape[:2]
        x1, y1 = cx_px - half, cy_px - half
        x2, y2 = x1 + ts,      y1 + ts
        pad_l = max(0, -x1);  pad_t = max(0, -y1)
        pad_r = max(0, x2-w); pad_b = max(0, y2-h)
        crop = minimap_bgr[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        if pad_l or pad_t or pad_r or pad_b:
            crop = cv2.copyMakeBorder(
                crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101
            )
        patch = crop.astype(np.float32)
        for c in range(3):
            ch = patch[:, :, c]
            ch[self._outside] = float(np.mean(ch[~self._outside]))
        return np.clip(patch, 0, 255).astype(np.uint8)

    def identify_at(
        self,
        minimap_bgr: np.ndarray,
        cx_px: float,
        cy_px: float,
        threshold: float = 0.30,
    ) -> Tuple[Optional[str], float]:
        """
        Search ±_SMALL_SR px around the blob centroid.  At each candidate
        position extract a TS×TS patch, apply the circular mask (ring and
        map-background corners → channel mean → zero NCC weight), then
        compare against the full-square portrait templates.
        Returns (agent_name, score) or (None, 0.0).
        """
        if not self.ready:
            return None, 0.0
        best_name:  Optional[str] = None
        best_score: float         = threshold - 1e-6
        sr   = self._SMALL_SR
        ts   = self.TEMPLATE_SIZE
        cx_i, cy_i = round(cx_px), round(cy_px)
        for dy in range(-sr, sr + 1):
            for dx in range(-sr, sr + 1):
                patch = self._masked_patch(minimap_bgr, cx_i + dx, cy_i + dy)
                for agent, templs in self._templates.items():
                    for tmpl in templs:
                        score = float(
                            cv2.matchTemplate(patch, tmpl, cv2.TM_CCOEFF_NORMED)[0, 0]
                        )
                        if score > best_score:
                            best_score = score
                            best_name  = agent
        return best_name, max(0.0, best_score)

    def all_scores_at(
        self,
        minimap_bgr: np.ndarray,
        cx_px: float,
        cy_px: float,
    ) -> List[Tuple[str, float]]:
        """Return the best NCC score for every agent at ±_SMALL_SR offsets,
        sorted descending.  Used for diagnostics."""
        if not self.ready:
            return []
        sr  = self._SMALL_SR
        best: Dict[str, float] = {a: -1.0 for a in self._templates}
        cx_i, cy_i = round(cx_px), round(cy_px)
        for dy in range(-sr, sr + 1):
            for dx in range(-sr, sr + 1):
                patch = self._masked_patch(minimap_bgr, cx_i + dx, cy_i + dy)
                for agent, templs in self._templates.items():
                    for tmpl in templs:
                        score = float(
                            cv2.matchTemplate(patch, tmpl, cv2.TM_CCOEFF_NORMED)[0, 0]
                        )
                        if score > best[agent]:
                            best[agent] = score
        return sorted(best.items(), key=lambda x: x[1], reverse=True)

    def identify_in_region(
        self,
        minimap_bgr: np.ndarray,
        cx_px: float,
        cy_px: float,
        search_r: int,
        threshold: float = 0.25,
    ) -> Tuple[Optional[str], float, float, float, List[Tuple[str, float]]]:
        """Probe ±search_r pixel offsets around the blob centroid using masked
        patches (ring and background pixels zeroed out via _masked_patch) and
        return (best_agent, best_score, refined_cx, refined_cy, all_scores).

        refined_cx/cy is the pixel offset that produced the highest NCC score,
        allowing blobs centred between two icons to snap to the true face.
        Using masked patches (not raw matchTemplate) ensures the team-colour
        ring does not bias the score toward agents with a matching ring hue.
        """
        if not self.ready:
            return None, 0.0, cx_px, cy_px, []

        # ── Portrait-quality guard ───────────────────────────────────────────
        # TM_CCOEFF_NORMED is numerically unstable when patch variance ≈ 0,
        # but even a non-degenerate dark background gradient can produce
        # spurious high NCC scores.  Real agent portrait icons are:
        #   • Bright  : mean face-pixel luminance ≥ 45
        #   • Varied  : max-min range across face pixels ≥ 80
        # Dark map backgrounds / screen edges fail at least one of these.
        # We check face-circle pixels only (ring pixels are mean-filled by
        # _masked_patch and contribute zero discriminating power).
        _cx_i = round(cx_px)
        _cy_i = round(cy_px)
        _center_patch = self._masked_patch(minimap_bgr, _cx_i, _cy_i)
        _face_pix = _center_patch[~self._outside].astype(np.float32)  # (N, 3)
        _face_std = float(np.mean(np.std(_face_pix, axis=0)))  # mean per-channel σ
        if (float(np.mean(_face_pix)) < 45.0
                or int(np.ptp(_face_pix)) < 80
                or _face_std < self.PORTRAIT_STD_MIN):
            return None, 0.0, cx_px, cy_px, [(a, 0.0) for a in self._templates]
        # ─────────────────────────────────────────────────────────────────────

        sr     = search_r
        cx_i   = _cx_i
        cy_i   = _cy_i
        best_name:  Optional[str] = None
        best_score: float         = threshold - 1e-6
        best_dx: int = 0
        best_dy: int = 0
        best_all: Dict[str, float] = {a: -1.0 for a in self._templates}

        for dy in range(-sr, sr + 1):
            for dx in range(-sr, sr + 1):
                patch = self._masked_patch(minimap_bgr, cx_i + dx, cy_i + dy)
                # Per-position portrait guard: same dual check for every offset
                # so NCC is never called on dark/flat positions within the
                # search window.
                _fp = patch[~self._outside].astype(np.float32)
                _fp_std = float(np.mean(np.std(_fp, axis=0)))
                if (float(np.mean(_fp)) < 45.0
                        or int(np.ptp(_fp)) < 80
                        or _fp_std < self.PORTRAIT_STD_MIN):
                    continue
                for agent, templs in self._templates.items():
                    for tmpl in templs:
                        score = float(
                            cv2.matchTemplate(patch, tmpl, cv2.TM_CCOEFF_NORMED)[0, 0]
                        )
                        if score > best_all[agent]:
                            best_all[agent] = score
                        if score > best_score:
                            best_score = score
                            best_name  = agent
                            best_dx, best_dy = dx, dy

        refined_cx = float(cx_i + best_dx)
        refined_cy = float(cy_i + best_dy)
        all_scores = sorted(best_all.items(), key=lambda x: x[1], reverse=True)
        return best_name, max(0.0, best_score), refined_cx, refined_cy, all_scores

    def find_all(
        self,
        minimap_gray: np.ndarray,
        threshold: float = 0.18,
        max_per_agent: int = 2,
    ) -> List[Tuple[str, int, int, float]]:
        """
        Slide each template across the full minimap (grayscale).
        Returns [(agent_name, cx_px, cy_px, score)] sorted desc by score,
        after greedy NMS that merges hits within _NMS_DIST pixels.
        Up to max_per_agent hits per template (handles mirror-pick games where
        both teams field the same agent — each team's icon appears once).
        """
        if not self.ready:
            return []
        half = self.TEMPLATE_SIZE // 2
        hits: List[Tuple[float, int, int, str]] = []
        for agent, templs in self._templates.items():
            # Take element-wise max across all crops so every sample gets a vote
            result = cv2.matchTemplate(minimap_gray, templs[0], cv2.TM_CCOEFF_NORMED)
            for tmpl in templs[1:]:
                result = np.maximum(result, cv2.matchTemplate(minimap_gray, tmpl, cv2.TM_CCOEFF_NORMED))
            result_copy = result.copy()
            for _ in range(max_per_agent):
                _, max_val, _, max_loc = cv2.minMaxLoc(result_copy)
                if float(max_val) < threshold:
                    break
                hits.append((
                    float(max_val),
                    int(max_loc[0]) + half,
                    int(max_loc[1]) + half,
                    agent,
                ))
                # Suppress this peak so the next iteration finds a distinct 2nd instance
                r = self._NMS_DIST
                x1 = max(0, max_loc[0] - r)
                y1 = max(0, max_loc[1] - r)
                x2 = min(result_copy.shape[1], max_loc[0] + r + 1)
                y2 = min(result_copy.shape[0], max_loc[1] + r + 1)
                result_copy[y1:y2, x1:x2] = -1.0
        # Sort best-first, then greedy NMS across all agent hits
        hits.sort(key=lambda x: -x[0])
        kept: List[Tuple[str, int, int, float]] = []
        occupied: List[Tuple[int, int]] = []
        for score, cx, cy, agent in hits:
            if all(math.hypot(cx - ox, cy - oy) >= self._NMS_DIST
                   for ox, oy in occupied):
                kept.append((agent, cx, cy, score))
                occupied.append((cx, cy))
        return kept

    def restrict_to_agents(self, names: List[str]) -> None:
        """Remove all templates for agents not in *names* (case-insensitive).

        Call once after __init__ to constrain matching to the agents actually
        present in the match.  This eliminates phantom agents (e.g. 'miks',
        'chamber') that would otherwise score weakly but still win when the
        true agent's template is absent or has a similar face crop.
        """
        keep = {n.lower() for n in names}
        removed = [k for k in list(self._templates) if k not in keep]
        for k in removed:
            del self._templates[k]
        if removed:
            print(f"[AgentTemplateMatcher] Restricted to {len(self._templates)} agents "
                  f"(dropped: {', '.join(sorted(removed))})")


# ─────────────────────────────────────────────────────────────────────────────
# HUD agent detector — reads player-card agent icons at match start
# ─────────────────────────────────────────────────────────────────────────────

class HUDAgentDetector:
    """
    Detects which agents are in the match by matching player-card HUD icons
    against the reference killfeed portrait images.

    The top-level player-card ROI is defined in settings.py; the agent icon
    occupies the leftmost ~14 %×46 % sub-region of each card.  NCC against
    single-file killfeed reference images (one PNG per agent) gives a clean
    roster with very few frames needed.

    Typical usage
    ─────────────
        detector = HUDAgentDetector("valorant_resources/agents_killfeed")
        cap = cv2.VideoCapture(vod_path)
        left_agents, right_agents = detector.detect_roster(cap, scan_end_s=60)
        tracker = MinimapTracker(...,
                                 known_left_agents=left_agents,
                                 known_right_agents=right_agents)
    """

    TEMPLATE_SIZE = 32   # px — NCC template size
    NCC_THRESHOLD = 0.35  # minimum score to accept a match

    # Player card ROI constants (must match settings.py)
    LEFT_TEAM_X       = 0.005
    RIGHT_TEAM_X      = 0.820
    PLAYER_CARD_W     = 0.175
    PLAYER_CARD_H     = 0.090
    LEFT_PLAYER_Y     = [0.505, 0.605, 0.705, 0.805, 0.905]
    RIGHT_PLAYER_Y    = [0.505, 0.605, 0.705, 0.805, 0.905]
    # Agent icon sub-region within the card (normalised to card size)
    # Left cards: portrait is top-left; right cards: portrait is top-right (mirrored)
    AGENT_ICON_XYWH_LEFT  = (0.00, 0.00, 0.19, 0.50)
    AGENT_ICON_XYWH_RIGHT = (0.82, 0.00, 0.18, 0.50)
    AGENT_ICON_XYWH       = (0.00, 0.00, 0.19, 0.50)  # legacy alias (left)

    def __init__(self, killfeed_dir: str) -> None:
        self._templates: Dict[str, np.ndarray] = {}
        if os.path.isdir(killfeed_dir):
            self._load_templates(killfeed_dir)

    @property
    def ready(self) -> bool:
        return len(self._templates) > 0

    def _load_templates(self, killfeed_dir: str) -> None:
        ts    = self.TEMPLATE_SIZE
        _EXTS = {".png", ".jpg", ".jpeg", ".webp"}
        n = 0
        for fname in sorted(os.listdir(killfeed_dir)):
            fpath = os.path.join(killfeed_dir, fname)
            if not os.path.isfile(fpath):
                continue
            if os.path.splitext(fname)[1].lower() not in _EXTS:
                continue
            name = os.path.splitext(fname)[0].lower()
            img  = _load_image_bgr(fpath)
            if img is not None:
                self._templates[name] = cv2.resize(img, (ts, ts))
                n += 1
        print(f"[HUDAgentDetector] Loaded {n} killfeed templates from {killfeed_dir}")

    def _extract_agent_icon(self, frame_bgr: np.ndarray,
                            card_x: float, card_y: float,
                            right_side: bool = False) -> Optional[np.ndarray]:
        """Extract the agent icon sub-region from one player card."""
        fh, fw = frame_bgr.shape[:2]
        cx  = int(card_x * fw)
        cy  = int(card_y * fh)
        cw  = int(self.PLAYER_CARD_W * fw)
        ch  = int(self.PLAYER_CARD_H * fh)
        card = frame_bgr[cy:cy + ch, cx:cx + cw]
        if card.size == 0:
            return None
        ax, ay, aw, ah = (self.AGENT_ICON_XYWH_RIGHT if right_side
                          else self.AGENT_ICON_XYWH_LEFT)
        ix = int(ax * cw); iy = int(ay * ch)
        iw = max(1, int(aw * cw)); ih = max(1, int(ah * ch))
        icon = card[iy:iy + ih, ix:ix + iw]
        return icon if icon.size > 0 else None

    def _match_icon(self, icon_bgr: np.ndarray) -> Tuple[Optional[str], float]:
        """Return (agent_name, score) or (None, 0.0)."""
        ts = self.TEMPLATE_SIZE
        try:
            resized = cv2.resize(icon_bgr, (ts, ts))
        except cv2.error:
            return None, 0.0
        best_name:  Optional[str] = None
        best_score: float         = self.NCC_THRESHOLD - 1e-6
        for name, tmpl in self._templates.items():
            score = float(cv2.matchTemplate(resized, tmpl, cv2.TM_CCOEFF_NORMED)[0, 0])
            if score > best_score:
                best_score = score
                best_name  = name
        return best_name, max(0.0, best_score)

    def detect_from_frame(self, frame_bgr: np.ndarray) -> Dict[str, Optional[str]]:
        """Return {slot: agent} for all 10 player slots in one frame.

        Right-team icons are horizontally flipped in the broadcast layout
        (the card mirrors the left-team layout), so the extracted crop is
        flipped before NCC matching to align with the killfeed reference images.
        """
        result: Dict[str, Optional[str]] = {}
        for side, team_x, player_ys, right_side in (
            ("left",  self.LEFT_TEAM_X,  self.LEFT_PLAYER_Y,  False),
            ("right", self.RIGHT_TEAM_X, self.RIGHT_PLAYER_Y, True),
        ):
            for i, py in enumerate(player_ys):
                icon = self._extract_agent_icon(frame_bgr, team_x, py,
                                                right_side=right_side)
                if icon is None:
                    result[f"{side}_{i+1}"] = None
                    continue
                # Right-side icons are already on the correct side; flip for NCC
                # alignment with the left-facing killfeed reference portraits
                if right_side:
                    icon = cv2.flip(icon, 1)
                name, _score = self._match_icon(icon)
                result[f"{side}_{i+1}"] = name
        return result

    def detect_roster(
        self,
        cap: "cv2.VideoCapture",
        scan_start_s: float = 5.0,
        scan_end_s:   float = 60.0,
        scan_stride_s: float = 5.0,
        min_votes: int = 2,
    ) -> Tuple[List[Optional[str]], List[Optional[str]]]:
        """Scan several frames and return (left_agents[5], right_agents[5]).

        Each element is the majority-vote agent name, or None if not detected
        with enough confidence.  Slots with None should be excluded from the
        MinimapTracker known_*_agents lists (don't restrict to None).
        """
        from collections import Counter  # avoid top-level import overhead
        left_votes:  List[Counter] = [Counter() for _ in range(5)]
        right_votes: List[Counter] = [Counter() for _ in range(5)]

        t = scan_start_s
        while t <= scan_end_s:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ret, frame = cap.read()
            if not ret:
                break
            dets = self.detect_from_frame(frame)
            for i in range(5):
                a = dets.get(f"left_{i+1}")
                if a:
                    left_votes[i][a] += 1
                a = dets.get(f"right_{i+1}")
                if a:
                    right_votes[i][a] += 1
            t += scan_stride_s

        def _pick(votes: Counter) -> Optional[str]:
            if not votes:
                return None
            top, count = votes.most_common(1)[0]
            return top if count >= min_votes else None

        left_agents  = [_pick(v) for v in left_votes]
        right_agents = [_pick(v) for v in right_votes]
        print(f"[HUDAgentDetector] Left  roster: {left_agents}")
        print(f"[HUDAgentDetector] Right roster: {right_agents}")
        return left_agents, right_agents


# ─────────────────────────────────────────────────────────────────────────────
# Spike template matcher
# ─────────────────────────────────────────────────────────────────────────────

class SpikeTemplateMatcher:
    """
    Matches spike icon templates against the minimap via grayscale NCC.

    The spike appears as a distinctive yellow hexagonal icon on the minimap
    when dropped on the floor or planted.  Template images must live in a
    directory (typically {agents_dir}/spike/) and should be 72×72 crops
    saved at 3× zoom (the same format produced by _save_blob_crop).

    The class is kept deliberately simple:
      • No circular mask — the spike icon is small enough that surrounding
        map pixels add only minor noise.
      • Grayscale NCC — sufficient because the spike shape is highly
        distinctive regardless of colour.
      • Agent-position suppression in the result map prevents the spike
        NCC from firing on agent portrait faces (which can appear yellowish).

    FP-prevention note
    ──────────────────
    This class is entirely separate from AgentTemplateMatcher and from the
    crop-save quality gates.  It cannot trigger or bypass any of the 9
    crop-save gates.  Agent tracking is unaffected by spike NCC results.
    """

    # Spike crops are saved at 3× zoom (same as agent crops).
    # Resize to TEMPLATE_SIZE for 1× NCC on the actual minimap.
    TEMPLATE_SIZE = 24   # px
    NCC_THRESHOLD = 0.45  # minimum NCC to accept a spike hit.
                          # 0.38 fired on map-corner textures (bottom-right FP);
                          # raised to 0.45 after spike template calibration.

    def __init__(self, spike_dir: str) -> None:
        self._templates: List[np.ndarray] = []  # list of (TS×TS) uint8 BGR
        if os.path.isdir(spike_dir):
            self._load_templates(spike_dir)

    @property
    def ready(self) -> bool:
        return len(self._templates) > 0

    def _load_templates(self, spike_dir: str) -> None:
        _EXTS = {".png", ".jpg", ".jpeg", ".webp"}
        ts = self.TEMPLATE_SIZE
        n = 0
        for fname in sorted(os.listdir(spike_dir)):
            fpath = os.path.join(spike_dir, fname)
            if os.path.isfile(fpath) and os.path.splitext(fname)[1].lower() in _EXTS:
                img = _load_image_bgr(fpath)
                if img is not None:
                    # Crops are 72×72 (3× zoom); downscale to native 1×
                    self._templates.append(cv2.resize(img, (ts, ts)))
                    n += 1
        if n:
            print(f"[SpikeTemplateMatcher] Loaded {n} spike template(s) from {spike_dir}")
        else:
            print(f"[SpikeTemplateMatcher] WARNING: no spike templates found in {spike_dir}")

    def find(
        self,
        minimap_bgr: np.ndarray,
        suppress_px: List[Tuple[float, float]],
        suppress_r: int = 16,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Slide all spike templates across the minimap (grayscale NCC) and
        return (cx_px, cy_px, score) of the best hit, or None.

        suppress_px  : (cx_px, cy_px) of every detected agent blob this frame.
                       A suppress_r-pixel radius around each agent is blanked
                       in the NCC result map so agent faces (which may appear
                       yellowish) do not produce false spike hits.
        suppress_r   : pixel radius to suppress around each agent (default 16,
                       slightly larger than the agent icon radius ≈11 px so
                       the ring + a small margin are both excluded).
        """
        if not self.ready or minimap_bgr is None or minimap_bgr.size == 0:
            return None

        ts   = self.TEMPLATE_SIZE
        half = ts // 2
        gray = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY)

        # Build merged NCC result (element-wise max across all templates)
        result: Optional[np.ndarray] = None
        for tmpl in self._templates:
            tmpl_g = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
            r = cv2.matchTemplate(gray, tmpl_g, cv2.TM_CCOEFF_NORMED)
            result = r if result is None else np.maximum(result, r)

        if result is None:
            return None

        # Blank agent positions in NCC result space.
        # NCC result[y, x] corresponds to template top-left at (x, y),
        # i.e. template centre at (x + half, y + half).
        # To suppress agent at (acx, acy): blank result[acy-half ± r, acx-half ± r].
        rh, rw = result.shape[:2]
        for acx, acy in suppress_px:
            rx = int(round(acx)) - half
            ry = int(round(acy)) - half
            x1 = max(0, rx - suppress_r);  y1 = max(0, ry - suppress_r)
            x2 = min(rw, rx + suppress_r + 1); y2 = min(rh, ry + suppress_r + 1)
            result[y1:y2, x1:x2] = -1.0

        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if float(max_val) < self.NCC_THRESHOLD:
            return None

        cx_px = float(max_loc[0]) + half
        cy_px = float(max_loc[1]) + half
        return cx_px, cy_px, float(max_val)


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

class SpikeTrack:
    """
    Tracks the spike's independent position on the minimap for frames when
    it is on the ground (dropped) or planted.

    When the spike is carried by a player the carrying BlobTrack.has_spike
    flag is set instead; no SpikeTrack entry is created for carried state.

    state values
    ─────────────
    "ground"  — spike dropped on floor, not yet planted.
    "planted" — spike is armed; position is now fixed.
    """

    __slots__ = ("positions", "state", "last_cx", "last_cy", "last_seen_ms")

    def __init__(
        self,
        cx_norm: float,
        cy_norm: float,
        t_ms: float,
        state: str = "ground",
    ) -> None:
        self.positions: List[Tuple[float, float, float]] = [(t_ms, cx_norm, cy_norm)]
        self.state:        str   = state  # "ground" | "planted"
        self.last_cx:      float = cx_norm
        self.last_cy:      float = cy_norm
        self.last_seen_ms: float = t_ms

    def update(self, cx_norm: float, cy_norm: float, t_ms: float) -> None:
        self.last_cx      = cx_norm
        self.last_cy      = cy_norm
        self.last_seen_ms = t_ms
        self.positions.append((t_ms, cx_norm, cy_norm))


# ─────────────────────────────────────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────────────────────────────────────

class BlobTrack:
    """Single player icon tracked across minimap frames within one round."""

    __slots__ = (
        "track_id", "team", "player_name", "agent_name", "agent_score",
        "positions",            # [(t_ms, x_norm, y_norm)]
        "alive", "last_cx", "last_cy", "last_seen_ms",
        "missing_count", "death_ms", "has_spike",
        "unidentified_frames",  # consecutive frames matched but agent still None
    )

    def __init__(
        self,
        track_id:   str,
        team:       str,
        cx:         float,
        cy:         float,
        t_ms:       float,
        agent_name: Optional[str] = None,
    ) -> None:
        self.track_id    = track_id
        self.team        = team
        self.player_name: Optional[str] = None
        self.agent_name  = agent_name
        self.agent_score: float = 0.0
        self.positions: List[Tuple[float, float, float]] = [(t_ms, cx, cy)]
        self.alive        = True
        self.last_cx      = cx
        self.last_cy      = cy
        self.last_seen_ms = t_ms
        self.missing_count = 0
        self.death_ms: Optional[float] = None
        self.has_spike    = False
        self.unidentified_frames: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Tracker
# ─────────────────────────────────────────────────────────────────────────────

class MinimapTracker:
    """
    Tracks all 10 player positions on the minimap throughout each round.

    Parameters
    ----------
    left_players / right_players : roster for each side (any case).
    left_color / right_color     : "teal" or "red" — must match killfeed.
    agents_dir                   : path to agent portrait images (PNG/WebP).
                                   If None or missing, agent matching is off.
    agent_map                    : optional pre-known agent→player mapping.
    """

    # ── Player icon HSV thresholds ────────────────────────────────────────────
    _TEAL_LO  = np.array([ 75,  50,  80], dtype=np.uint8)
    _TEAL_HI  = np.array([ 95, 145, 255], dtype=np.uint8)
    # Spike glow can shift a teal ring into the blue range; catch it separately
    # with a raised saturation floor (S≥90) to avoid grey map-background noise.
    _TEAL_LO2 = np.array([ 95,  90,  80], dtype=np.uint8)
    _TEAL_HI2 = np.array([130, 200, 255], dtype=np.uint8)
    _RED_LO1  = np.array([  0,  75,  95], dtype=np.uint8)
    _RED_HI1  = np.array([ 15, 220, 195], dtype=np.uint8)
    _RED_LO2  = np.array([163,  75,  95], dtype=np.uint8)
    _RED_HI2  = np.array([179, 220, 195], dtype=np.uint8)
    # Tighter red ranges used ONLY for the ability-icon centre check.
    # Skin-tone pixels (H≈0-15, S≈60-120) overlap the broad ring range above;
    # ability icon fills have highly-saturated ring colour (S ≥ 130).
    _CTR_RED_LO1 = np.array([  0, 130,  80], dtype=np.uint8)
    _CTR_RED_HI1 = np.array([ 15, 255, 220], dtype=np.uint8)
    _CTR_RED_LO2 = np.array([163, 130,  80], dtype=np.uint8)
    _CTR_RED_HI2 = np.array([179, 255, 220], dtype=np.uint8)

    # ── Spike yellow HSV (carrier / floor / planted — always the same colour)
    _SPIKE_LO = np.array([ 15, 130, 140], dtype=np.uint8)
    _SPIKE_HI = np.array([ 40, 255, 255], dtype=np.uint8)

    # ── Player icon geometry ──────────────────────────────────────────────────
    MIN_AREA   =  40    # px² (lowered from 80 to catch spike-occluded rings)
    MAX_AREA   = 2200   # px²
    MIN_RADIUS =   5    # px
    MAX_RADIUS =  30    # px

    # ── Spike geometry ────────────────────────────────────────────────────────
    SPIKE_MIN_AREA         =  10    # px²
    SPIKE_MAX_AREA         = 600    # px²
    SPIKE_CARRIER_MAX_DIST = 0.12   # normalised; ~58 px on 480-wide minimap

    # ── Player icon shape filters ─────────────────────────────────────────────
    # Player icons are near-circular ring arcs; barriers and ability dots are
    # solid shapes.  Two complementary filters:
    #
    #   1. Aspect-ratio: rejects elongated barriers (40×6 → ratio 6.7) and
    #      ability lines.
    #   2. Solidity (area / convex_hull_area): player ring arcs are hollow, so
    #      their pixel area is much less than the convex hull area
    #      (solidity ≈ 0.45–0.85).  Solid barriers and ability dots have
    #      solidity ≈ 0.95–1.0 and are rejected.
    MAX_ASPECT_RATIO  = 3.5
    MAX_BLOB_SOLIDITY = 0.90   # above this → solid shape (barrier / ability dot)
    BORDER_MARGIN_PX  = 25    # blobs this close to the minimap edge are UI artefacts
    # Blobs above this area (px²) may be two touching icons; attempt to split
    # them into individual circles via a second HoughCircles pass.
    SPLIT_AREA_THRESHOLD = 280
    # Portrait-centre texture: agent icons have a face inside the ring (high
    # brightness variance).  Hollow ability indicators have a flat/empty centre.
    # If the V-channel std-dev of the centre 15×15 px region is below this
    # value the blob is rejected as a non-portrait ring.
    MIN_PORTRAIT_STDDEV = 12.0
    # Ability icons fill their entire interior with the same colour as the ring;
    # real agent portraits have very few ring-colour pixels in the centre.
    # Real agents measured at 0–32%; ability icons at 87–100%.
    # Threshold of 0.40 leaves a large margin on both sides.
    MAX_CTR_TEAL_FRAC  = 0.40

    # ── Agent identification at confirmed blob positions ───────────────────────
    FACE_CROP_NCC_THRESHOLD  = 0.50  # NCC required to name an agent from its blob
    CROP_SAVE_NCC_THRESHOLD  = 0.70  # higher bar for saving harvest crops — filters
                                     # terrain textures that pass the portrait guard
                                     # but produce spuriously moderate NCC scores
    CROP_SAVE_MIN_MARGIN     = 0.12  # best_score − 2nd_best_score must exceed this
                                     # before a crop is saved.  Background textures
                                     # match multiple agents similarly (gap ≈ 0.02–
                                     # 0.08); genuine icons have one clear winner
                                     # (gap ≈ 0.25–0.45).  Filters false positives
                                     # that pass the NCC threshold by accident.
    CROP_SAVE_MIN_RING_FRAC  = 0.08  # min fraction of ring pixels (r=8–11 in 24×24
                                     # template space) that must be orange/red
                                     # (H≤15 or H≥160, S>50, V>50) or teal/green
                                     # (H=40–100, S>50, V>50) to save a crop.
                                     # Real icons always ≥0.24; background patches
                                     # (gray corners, blue-gray terrain) score
                                     # 0.00–0.07.  Edge crops are exempt.
    CROP_SAVE_MIN_FACE_STD   = 20.0  # min mean per-channel std-dev of face pixels
                                     # in the raw 24×24 crop before saving.
                                     # Gray-corner false positives score 18–21;
                                     # real agent faces score ≥20 (Astra min=20).
                                     # Separate from PORTRAIT_STD_MIN (detection).
    CROP_SAVE_STRICT_RING_THRESH = 0.35  # combined gate: if ring_frac < this AND
    CROP_SAVE_STRICT_FACE_STD    = 23.0  # face_std < this → reject.  Catches gray-
                                         # corner FPs that slip through both single
                                         # gates (e.g. rfrac≈0.33, fstd≈21).
                                         # Astra safe (rfrac≥0.47); waylay safe
                                         # (rfrac≈0.24 but fstd≥24).
    CROP_SAVE_MAX_FACE_TEAL_FRAC = 0.80  # teal map elements score 0.91–1.00;
                                         # real max: Vyse=0.65, Harbor=0.46.
    CROP_SAVE_MAX_FACE_GRAY_FRAC = 0.80  # gray map corners score 0.95;
                                         # real max: <0.60.
    CROP_SAVE_MIN_FACE_HUE_STD   = 10.0  # solid-color map elements (purple door,
                                         # brown panel, dark-red wall) have uniform
                                         # hue → std<6.  Real faces (incl. Omen)
                                         # score ≥14.  Threshold=10 is safe.
    CROP_SAVE_DEAD_RING_RFRAC    = 0.20  # dead-ring gate: if ring_frac < this …
    CROP_SAVE_DEAD_RING_RSAT     = 50.0  # … AND mean ring saturation < this,
                                         # there is no real agent ring → reject.
    # ── Additional face / ring gates (rounds 2–3) ─────────────────────────────
    CROP_SAVE_MIN_FACE_V         = 78.0  # Gate 5: very dark face → map element.
                                         # FP floor 73.7; TP min 87.7 → safe margin.
    CROP_SAVE_DEAD_RING_RFRAC2   = 0.22  # Gate 6: tighter dead-ring pair (catches
    CROP_SAVE_DEAD_RING_RSAT2    = 63.0  # FPs with rfrac=0.22, rsat=62).
    CROP_SAVE_DEAD_RING_RFRAC3   = 0.30  # Gate 7: broader rfrac dead-ring (catches
    CROP_SAVE_DEAD_RING_RSAT3    = 55.0  # FPs with rfrac=0.28, rsat=53).
    CROP_SAVE_LOW_HUE_STD        = 12.0  # Gate 8: red face + barely-nonuniform hue
    CROP_SAVE_LOW_HUE_FRED_MIN   = 0.45  # → map element.  FP: (10.7, 0.56).
    CROP_SAVE_PURPLE_FRED_MIN    = 0.35  # Gate 9: dark purple+red face (ability icon).
    CROP_SAVE_PURPLE_MIN         = 0.30  # FP: (fpurple=0.35, fred=0.43, fv=91.6).
    CROP_SAVE_PURPLE_V_MAX       = 100.0

    # ── Ring sampling (used only for the blob-first path) ─────────────────────
    RING_SAMPLE_INNER   = 11     # px — inner edge of team-ring sampling annulus
    RING_SAMPLE_OUTER   = 20     # px — outer edge of team-ring sampling annulus
    RING_MIN_PIXELS     =  3     # ring pixels required to call a team

    # ── Tracking ─────────────────────────────────────────────────────────────
    MAX_PLAYERS          = 10   # blobs per team passed to tracker. Real max is 5,
                                # but raising this ensures no real player is ever
                                # capped out.  Ability-icon false positives are
                                # cleaned up by MAX_NO_AGENT_FRAMES.
    MAX_ASSIGN_DIST      = 0.22    # normalised; ~105 px on 480-wide minimap
    DEAD_AFTER_N         = 4       # consecutive update cycles missing → dead
    MAX_NO_AGENT_FRAMES  = 40      # expire tracks that never get NCC match (e.g. ability markers)
                                # Raised so real-agent tracks with weak NCC scores
                                # (due to template mismatch) survive long enough
                                # to contribute crops for re-labeling.

    # ── Colour-blob fallback geometry (used when no agent templates) ─────────
    MIN_AREA   =  40    # px² (lowered from 80 to catch spike-occluded rings)
    MAX_AREA   = 2200   # px²
    MIN_RADIUS =   5    # px
    MAX_RADIUS =  30    # px
    _KERNEL     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    _KERNEL_RED = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def __init__(
        self,
        left_players:  List[str],
        right_players: List[str],
        left_color:  str = "teal",
        right_color: str = "red",
        agents_dir:  Optional[str] = None,
        agent_map:   Optional[Dict[str, str]] = None,
        crop_dump_dir: Optional[str] = None,
        known_left_agents:  Optional[List[str]] = None,
        known_right_agents: Optional[List[str]] = None,
    ) -> None:
        self.left_players  = [p.lower() for p in left_players]
        self.right_players = [p.lower() for p in right_players]
        self.left_color    = left_color
        self.right_color   = right_color

        # Per-team known agent sets (None = unconstrained)
        self._known_left_agents:  Optional[set] = (
            {a.lower() for a in known_left_agents}  if known_left_agents  else None
        )
        self._known_right_agents: Optional[set] = (
            {a.lower() for a in known_right_agents} if known_right_agents else None
        )

        # Agent template matcher (optional)
        self._agent_matcher: Optional[AgentTemplateMatcher] = None
        if agents_dir:
            m = AgentTemplateMatcher(agents_dir)
            if m.ready:
                self._agent_matcher = m
                # Restrict templates to the union of both known rosters so
                # off-roster agents (e.g. 'miks', 'chamber') can never match.
                _all_known = (
                    (self._known_left_agents  or set()) |
                    (self._known_right_agents or set())
                )
                if _all_known:
                    self._agent_matcher.restrict_to_agents(list(_all_known))

        # Spike template matcher (optional — loaded from {agents_dir}/spike/)
        # Uses NCC to locate the spike on the ground/planted independently
        # from agent tracking.  Completely separate code path; does NOT
        # interact with the 9 crop-save quality gates.
        self._spike_matcher: Optional[SpikeTemplateMatcher] = None
        if agents_dir:
            _spike_subdir = os.path.join(agents_dir, "spike")
            if os.path.isdir(_spike_subdir):
                sm = SpikeTemplateMatcher(_spike_subdir)
                if sm.ready:
                    self._spike_matcher = sm

        # Cross-round agent ↔ player mappings (persist across reset_round)
        self._agent_to_player: Dict[str, str] = {
            a.lower(): p.lower() for a, p in (agent_map or {}).items()
        }
        self._player_to_agent: Dict[str, str] = {
            v: k for k, v in self._agent_to_player.items()
        }
        # Team-specific reverse maps (populated by set_player_agent_map)
        self._left_agent_to_player:  Dict[str, str] = {}
        self._right_agent_to_player: Dict[str, str] = {}

        # Per-round state (cleared by reset_round)
        self._tracks: Dict[str, BlobTrack] = {}
        self._next_idx: Dict[str, int] = {"left": 0, "right": 0}
        self._name_to_track: Dict[str, str] = {}
        self._last_positions: Dict[str, Tuple[float, float, float]] = {}
        self._round_number = 0
        self._event_buffer: List[dict] = []
        self._spike_track: Optional[SpikeTrack] = None  # independent spike position
        self._spike_planted: bool = False                # set by notify_spike_planted

        # Crop harvesting (optional — set crop_dump_dir to enable)
        self._crop_dump_dir: Optional[str] = crop_dump_dir
        if crop_dump_dir:
            os.makedirs(crop_dump_dir, exist_ok=True)
        self._frame_count: int = 0
        # Save every Nth frame to avoid dumping near-identical consecutive frames.
        # Lower = more crops; raise for very long videos to keep disk usage down.
        self._crop_save_stride: int = 3
        # Skip saving a crop when another blob centre is within this many px.
        # 15px ≈ half an icon diameter — pairs tighter than this produce crops
        # that are too overlapping to label reliably.
        self._crop_min_sep_px: float = 15.0

    # ── Crop harvesting ───────────────────────────────────────────────────────

    def _save_blob_crop(
        self,
        minimap_bgr: np.ndarray,
        cx_px: float,
        cy_px: float,
        tag: str,
    ) -> None:
        """Save a raw (unmasked) face crop for manual labeling.

        Crops are saved as {frame:06d}_{tag}_{cx}_{cy}.png at 3× zoom
        (INTER_NEAREST) so individual pixels are easy to see when labeling.
        Drop each file into a subfolder named after the agent, then run
        build_templates_from_crops.py to produce averaged templates.
        """
        ts   = self._agent_matcher.TEMPLATE_SIZE if self._agent_matcher else 24
        half = ts // 2
        h, w = minimap_bgr.shape[:2]
        cx, cy = round(cx_px), round(cy_px)
        x1, y1 = cx - half, cy - half
        x2, y2 = x1 + ts,   y1 + ts
        pad_l = max(0, -x1);  pad_t = max(0, -y1)
        pad_r = max(0, x2-w); pad_b = max(0, y2-h)
        crop = minimap_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if pad_l or pad_t or pad_r or pad_b:
            crop = cv2.copyMakeBorder(
                crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101
            )
        big  = cv2.resize(crop, (ts * 3, ts * 3), interpolation=cv2.INTER_NEAREST)
        fname = f"{self._frame_count:06d}_{tag}_{cx}_{cy}.png"
        cv2.imwrite(os.path.join(self._crop_dump_dir, fname), big)

    # ── Public API ────────────────────────────────────────────────────────────

    def reset_round(self, round_number: int) -> None:
        """Reset per-round state.  Cross-round agent mappings are preserved."""
        self._tracks.clear()
        self._next_idx = {"left": 0, "right": 0}
        self._name_to_track.clear()
        self._last_positions.clear()
        self._round_number = round_number
        self._spike_track   = None
        self._spike_planted = False

    def apply_roster(
        self,
        left_agents:  List[str],
        right_agents: List[str],
    ) -> None:
        """Restrict agent NCC matching to the 10 agents actually in the match.

        Called once from the processing pipeline after the HUD roster scan
        completes (~30 s into the match).  Updates per-team known-agent sets
        and prunes the AgentTemplateMatcher template dictionary so off-roster
        agents (e.g. 'miks', 'chamber') can never win an NCC comparison.
        """
        self._known_left_agents  = {a.lower() for a in left_agents}  if left_agents  else None
        self._known_right_agents = {a.lower() for a in right_agents} if right_agents else None
        if self._agent_matcher:
            _all_known = (
                (self._known_left_agents  or set()) |
                (self._known_right_agents or set())
            )
            if _all_known:
                self._agent_matcher.restrict_to_agents(list(_all_known))
        print(
            f"[MinimapTracker] Roster applied — "
            f"left: {sorted(self._known_left_agents or [])}  "
            f"right: {sorted(self._known_right_agents or [])}"
        )

    def set_player_agent_map(self, player_to_agent: Dict[str, str]) -> None:
        """Set a direct player → agent binding from the HUD card OCR scan.

        Derived from running NCC on agent_icon and OCR on player_name for
        each player card slot simultaneously at ~30 s into the match.
        This is the authoritative mapping used to annotate POSITION_UPDATE
        events with player names rather than relying on track fragmentation.

        Args:
            player_to_agent: {canonical_player_name: agent_name} for all 10 players.
        """
        self._agent_to_player = {
            agent.lower(): player for player, agent in player_to_agent.items()
        }
        self._player_to_agent = {
            player: agent.lower() for player, agent in player_to_agent.items()
        }
        # Team-specific reverse maps — disambiguate when both teams share an agent name.
        self._left_agent_to_player = {
            agent.lower(): player
            for player, agent in player_to_agent.items()
            if player.lower() in self.left_players
        }
        self._right_agent_to_player = {
            agent.lower(): player
            for player, agent in player_to_agent.items()
            if player.lower() in self.right_players
        }
        print(f"[MinimapTracker] Player→agent map set from HUD scan: {player_to_agent}")

    def update(
        self,
        minimap_bgr: np.ndarray,
        t_ms: float,
        round_number: int,
    ) -> List[dict]:
        """
        Process one minimap frame.
        Returns a list containing one POSITION_UPDATE event (or empty).
        """
        if minimap_bgr is None or minimap_bgr.size == 0:
            return []

        h, w = minimap_bgr.shape[:2]
        hsv  = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)

        # ── Player detection ─────────────────────────────────────────────────
        # Blob-first: find circular ring blobs → confirm player positions.
        # Circularity filter rejects pre-round barriers and ability orbs.
        # Agent identity is looked up at each confirmed position via NCC.
        #
        # Sort by area descending so real player rings (larger) beat ability
        # orbs (smaller) for the MAX_PLAYERS cap.  Crops are saved from the
        # FULL pre-cap list so ability markers are captured for inspection too.
        left_blobs_all  = sorted(self._find_blobs(hsv, self.left_color,  h, w),
                                 key=lambda b: -b[2])
        right_blobs_all = sorted(self._find_blobs(hsv, self.right_color, h, w),
                                 key=lambda b: -b[2])
        left_blobs  = left_blobs_all[:self.MAX_PLAYERS]
        right_blobs = right_blobs_all[:self.MAX_PLAYERS]

        # All blob positions from both teams (full list), used to detect
        # overlapping pairs when saving crops.
        all_blob_px = [
            (b[4], b[5]) for b in left_blobs_all + right_blobs_all
        ]

        self._frame_count += 1
        _img_h, _img_w = minimap_bgr.shape[:2]

        # Pre-compute spike mask and all blob positions once per frame so
        # _dets_from_blobs can use them for neighbour-blanking and spike discount.
        _spike_mask       = cv2.inRange(hsv, self._SPIKE_LO, self._SPIKE_HI)
        _all_blob_px      = [(b[4], b[5]) for b in left_blobs + right_blobs]
        _face_r           = (self._agent_matcher.TEMPLATE_SIZE // 2
                             if self._agent_matcher else 12)

        def _dets_from_blobs(blobs, team_tag: str):
            dets = []
            for idx, (cx_n, cy_n, _area, _r, cx_px, cy_px) in enumerate(blobs):
                if self._agent_matcher and self._agent_matcher.ready:
                    # Search radius scales with blob size so the NCC window
                    # covers the full extent of merged blobs and can snap to
                    # the true face centre rather than the arc centroid.
                    _sr = min(max(self._agent_matcher._SMALL_SR, int(_r * 0.45)), 10)

                    # ── Neighbour-blanking ────────────────────────────────────
                    # Replace each nearby blob's face circle with its local mean
                    # so the NCC sees only the target agent's face pixels.
                    # Only blank pixels that are inside the neighbour's circle
                    # but NOT inside the target's circle — this preserves all
                    # of the target's own face pixels even when the two icons
                    # overlap heavily.
                    mm_clean = minimap_bgr.copy()
                    _cx_i, _cy_i = int(cx_px), int(cy_px)
                    for _nb_cx, _nb_cy in _all_blob_px:
                        _d = math.hypot(cx_px - _nb_cx, cy_px - _nb_cy)
                        if _d < 2 or _d > _face_r * 2:
                            continue  # skip self and blobs too far to overlap
                        _nx, _ny = int(_nb_cx), int(_nb_cy)
                        _ny1 = max(0, _ny - _face_r)
                        _ny2 = min(_img_h, _ny + _face_r + 1)
                        _nx1 = max(0, _nx - _face_r)
                        _nx2 = min(_img_w, _nx + _face_r + 1)
                        _reg = mm_clean[_ny1:_ny2, _nx1:_nx2].astype(np.float32)
                        _ys, _xs = np.mgrid[0:_ny2 - _ny1, 0:_nx2 - _nx1]
                        # Pixels in neighbour's circle
                        _ins_nb = ((_xs - (_nx - _nx1)) ** 2
                                   + (_ys - (_ny - _ny1)) ** 2) <= _face_r ** 2
                        # Pixels in target's circle (in the same grid coords)
                        _ins_tgt = ((_xs - (_cx_i - _nx1)) ** 2
                                    + (_ys - (_cy_i - _ny1)) ** 2) <= _face_r ** 2
                        # Only blank the exclusive part of the neighbour —
                        # preserves all of the target's own face pixels even
                        # when the two icons overlap heavily.
                        _ins = _ins_nb & ~_ins_tgt
                        if _ins.any():
                            for _c in range(3):
                                _ch = _reg[:, :, _c]
                                _ch[_ins] = float(np.mean(_ch[_ins_nb] if _ins_nb.any()
                                                           else _ch))
                        mm_clean[_ny1:_ny2, _nx1:_nx2] = (
                            _reg.clip(0, 255).astype(np.uint8)
                        )

                    # ── Spike-pixel removal ───────────────────────────────────
                    # Yellow spike pixels that overlap the face crop bias NCC.
                    # Replace them with the mean of the non-spike face pixels.
                    _fy1 = max(0, _cy_i - _face_r)
                    _fy2 = min(_img_h, _cy_i + _face_r + 1)
                    _fx1 = max(0, _cx_i - _face_r)
                    _fx2 = min(_img_w, _cx_i + _face_r + 1)
                    _reg_sp = mm_clean[_fy1:_fy2, _fx1:_fx2].astype(np.float32)
                    _is_spike = _spike_mask[_fy1:_fy2, _fx1:_fx2] > 0
                    if _is_spike.any():
                        _not_spike = ~_is_spike
                        for _c in range(3):
                            _ch = _reg_sp[:, :, _c]
                            _fill = (float(np.mean(_ch[_not_spike]))
                                     if _not_spike.any() else float(np.mean(_ch)))
                            _ch[_is_spike] = _fill
                        mm_clean[_fy1:_fy2, _fx1:_fx2] = (
                            _reg_sp.clip(0, 255).astype(np.uint8)
                        )

                    # ── Spike-proximity threshold discount ────────────────────
                    # Even after spike-pixel removal some residual score loss
                    # occurs; accept a slightly lower threshold when the spike
                    # overlapped this icon's crop.
                    _ncc_thresh = (
                        max(0.14, self.FACE_CROP_NCC_THRESHOLD - 0.08)
                        if _is_spike.any()
                           and float(np.count_nonzero(_is_spike)) / _is_spike.size > 0.10
                        else self.FACE_CROP_NCC_THRESHOLD
                    )

                    agent, score, ref_cx, ref_cy, all_scores = (
                        self._agent_matcher.identify_in_region(
                            mm_clean, cx_px, cy_px, _sr,
                            threshold=_ncc_thresh,
                        )
                    )
                    # Use refined position for tracking accuracy
                    cx_px = ref_cx
                    cy_px = ref_cy
                    cx_n  = cx_px / _img_w
                    cy_n  = cy_px / _img_h
                    # Save crop only when NCC confirms a valid agent identity
                    # with high confidence AND clear discrimination:
                    #   • score >= CROP_SAVE_NCC_THRESHOLD (0.70): absolute floor
                    #   • score − 2nd_best >= CROP_SAVE_MIN_MARGIN (0.12): one agent
                    #     must clearly win.  Background textures match all agents
                    #     similarly (low margin) and are rejected by this gate even
                    #     when the absolute score sneaks past the threshold.
                    _2nd_score = all_scores[1][1] if len(all_scores) >= 2 else 0.0
                    _margin    = score - _2nd_score
                    if (agent is not None
                            and score >= self.CROP_SAVE_NCC_THRESHOLD
                            and _margin >= self.CROP_SAVE_MIN_MARGIN
                            and self._crop_dump_dir
                            and (self._frame_count - 1) % self._crop_save_stride == 0):
                        # ── Ring team-colour gate ─────────────────────────────
                        # The annulus r=8-11 from the crop centre must contain
                        # ≥CROP_SAVE_MIN_RING_FRAC orange or teal/green pixels.
                        # Background terrain always fails (≈0); real icons pass.
                        # Crops at the minimap edge (ring off-screen) are exempt.
                        _ts  = self._agent_matcher.TEMPLATE_SIZE    # 24
                        _rf  = self._agent_matcher.FACE_MASK_RADIUS  # 8
                        _hlf = _ts // 2
                        _cxi, _cyi = round(cx_px), round(cy_px)
                        _hmm, _wmm = minimap_bgr.shape[:2]
                        _x1r = _cxi - _hlf;  _y1r = _cyi - _hlf
                        if (_x1r >= 0 and _y1r >= 0
                                and _x1r + _ts <= _wmm
                                and _y1r + _ts <= _hmm):
                            _raw_r    = minimap_bgr[_y1r:_y1r+_ts, _x1r:_x1r+_ts]
                            _ys_r, _xs_r = np.mgrid[0:_ts, 0:_ts]
                            _r2_r     = (_xs_r - _hlf)**2 + (_ys_r - _hlf)**2
                            # Face std gate (before ring check)
                            _face_px_r  = _raw_r[_r2_r <= _rf**2]
                            _face_std_r = float(np.mean(
                                np.std(_face_px_r.astype(np.float32), axis=0)
                            ))
                            # Face HSV — teal / gray / hue-uniformity gates
                            _fhsv       = cv2.cvtColor(
                                _face_px_r.reshape(1, -1, 3), cv2.COLOR_BGR2HSV
                            )[0]
                            _fH = _fhsv[:, 0].astype(np.int32)
                            _fS = _fhsv[:, 1].astype(np.int32)
                            _fV = _fhsv[:, 2].astype(np.int32)
                            _face_teal_frac = float(
                                ((_fH >= 40) & (_fH <= 100) & (_fS > 40)).sum()
                            ) / len(_fH)
                            _face_gray_frac = float((_fS < 30).sum()) / len(_fH)
                            _fsat_mask  = _fS > 30
                            _face_hue_std = (
                                float(np.std(_fH[_fsat_mask]))
                                if _fsat_mask.sum() > 5 else 0.0
                            )
                            _face_mean_v    = float(np.mean(_fV))
                            _face_red_frac  = float(
                                (((_fH <= 15) | (_fH >= 160)) & (_fS > 40)).sum()
                            ) / len(_fH)
                            _face_purple_frac = float(
                                ((_fH >= 120) & (_fH <= 160) & (_fS > 40)).sum()
                            ) / len(_fH)
                            _ring_px  = _raw_r[(_r2_r > _rf**2) & (_r2_r <= 11**2)]
                            _rhsv     = cv2.cvtColor(
                                _ring_px.reshape(1, -1, 3), cv2.COLOR_BGR2HSV
                            )[0]
                            _Rh, _Rs, _Rv = _rhsv[:, 0], _rhsv[:, 1], _rhsv[:, 2]
                            _ring_frac = float(
                                (
                                    ((_Rh <= 15) | (_Rh >= 160))
                                    & (_Rs > 50) & (_Rv > 50)    # orange/red
                                    | ((_Rh >= 40) & (_Rh <= 100))
                                    & (_Rs > 50) & (_Rv > 50)    # teal/green
                                ).sum()
                            ) / len(_Rh)
                            _rsat_mean = float(np.mean(_Rs))
                            _ring_ok = (
                                _ring_frac >= self.CROP_SAVE_MIN_RING_FRAC
                                and _face_std_r >= self.CROP_SAVE_MIN_FACE_STD
                                # combined gate: low-ring crops must be textured
                                and (
                                    _ring_frac >= self.CROP_SAVE_STRICT_RING_THRESH
                                    or _face_std_r >= self.CROP_SAVE_STRICT_FACE_STD
                                )
                                # teal map elements (fteal=0.91–1.00)
                                and _face_teal_frac <= self.CROP_SAVE_MAX_FACE_TEAL_FRAC
                                # gray map corners (fgray=0.95)
                                and _face_gray_frac <= self.CROP_SAVE_MAX_FACE_GRAY_FRAC
                                # solid-color elements: hue is nearly uniform
                                and _face_hue_std >= self.CROP_SAVE_MIN_FACE_HUE_STD
                                # dead-ring: low ring fraction + desaturated ring
                                and not (
                                    _ring_frac < self.CROP_SAVE_DEAD_RING_RFRAC
                                    and _rsat_mean < self.CROP_SAVE_DEAD_RING_RSAT
                                )
                                # Gate 5: very dark face → map element
                                and _face_mean_v >= self.CROP_SAVE_MIN_FACE_V
                                # Gate 6: tighter dead-ring pair
                                and not (
                                    _ring_frac < self.CROP_SAVE_DEAD_RING_RFRAC2
                                    and _rsat_mean < self.CROP_SAVE_DEAD_RING_RSAT2
                                )
                                # Gate 7: broader rfrac dead-ring
                                and not (
                                    _ring_frac < self.CROP_SAVE_DEAD_RING_RFRAC3
                                    and _rsat_mean < self.CROP_SAVE_DEAD_RING_RSAT3
                                )
                                # Gate 8: red face + barely-nonuniform hue
                                and not (
                                    _face_hue_std < self.CROP_SAVE_LOW_HUE_STD
                                    and _face_red_frac > self.CROP_SAVE_LOW_HUE_FRED_MIN
                                )
                                # Gate 9: dark purple+red face (ability icon)
                                and not (
                                    _face_purple_frac > self.CROP_SAVE_PURPLE_MIN
                                    and _face_red_frac > self.CROP_SAVE_PURPLE_FRED_MIN
                                    and _face_mean_v < self.CROP_SAVE_PURPLE_V_MAX
                                )
                            )
                        else:
                            _ring_ok = True  # edge crop — ring off-screen, exempt
                        if _ring_ok:
                            self._save_blob_crop(minimap_bgr, cx_px, cy_px,
                                                 f"{team_tag}{idx}_{agent}_ncc{score:.4f}_t{int(t_ms/1000)}s")
                else:
                    all_scores = []
                    agent, score = None, 0.0
                dets.append((cx_n, cy_n, cx_px, cy_px, agent, score, all_scores))
            return dets

        left_dets  = _dets_from_blobs(left_blobs,  "L")

        # ── Cross-team blob decontamination ───────────────────────────────────
        # A face-portrait's skin tones can trigger the opposite team's colour
        # mask (e.g. Breach's orange face → red mask).  Drop any right-team blob
        # whose centre is within 8 px of a left-team blob centre, and vice-versa.
        _CONTAM_R2 = (8.0 / w) ** 2   # normalised-space threshold
        def _filter_blobs(src_blobs, ref_blobs):
            filtered = []
            for b in src_blobs:
                bcx, bcy = b[0], b[1]
                if any((bcx - r[0]) ** 2 + (bcy - r[1]) ** 2 < _CONTAM_R2
                       for r in ref_blobs):
                    continue
                filtered.append(b)
            return filtered
        right_blobs = _filter_blobs(right_blobs, left_blobs)
        left_blobs  = _filter_blobs(left_blobs,  right_blobs)

        right_dets = _dets_from_blobs(right_blobs, "R")

        self._update_team_tracks(left_dets,  "left",  t_ms)
        self._update_team_tracks(right_dets, "right", t_ms)
        self._deduplicate_agents_across_teams()

        # ── Spike detection (HSV blobs + optional NCC refinement) ────────────
        # Step 1: HSV colour-blob detection.  Reliable for both carried spike
        # (yellow blob near a player → has_spike=True) and loose/planted spike
        # (isolated yellow blob → spike_loose).
        spike_blobs = self._find_spike_blobs(hsv, h, w)

        # Step 2 (optional): NCC spike template matching with agent-position
        # suppression.  Adds precision for ground/planted spike when the HSV
        # blob centroid is imprecise or when the spike is partially occluded.
        # IMPORTANT: this is a completely independent code path — it never
        # touches the agent NCC loop, the 9 crop-save gates, or agent tracks.
        if self._spike_matcher and self._spike_matcher.ready:
            _ncc_spike = self._spike_matcher.find(
                minimap_bgr, suppress_px=_all_blob_px
            )
            if _ncc_spike is not None:
                _snx_px, _sny_px, _sncc = _ncc_spike
                _snx_n,  _sny_n  = _snx_px / w, _sny_px / h
                # Only add if there is no existing HSV blob within 6 % of the
                # minimap width at this position (avoids duplicate near-same hits).
                _MERGE_D = 0.06
                if not any(
                    math.hypot(_snx_n - sb[0], _sny_n - sb[1]) < _MERGE_D
                    for sb in spike_blobs
                ):
                    spike_blobs.append((_snx_n, _sny_n, 5.0))  # area=5 sentinel

        spike_loose = self._assign_spike(spike_blobs, t_ms)

        # ── Update last-known positions for named tracks ──────────────────────
        for track in self._tracks.values():
            if track.alive and track.player_name:
                self._last_positions[track.player_name] = (
                    track.last_cx, track.last_cy, t_ms
                )

        # ── Build event ───────────────────────────────────────────────────────
        positions = []
        for track in self._tracks.values():
            if not track.alive:
                continue
            entry: dict = {
                "track_id":   track.track_id,
                "player":     track.player_name or f"unknown_{track.track_id}",
                "agent":      track.agent_name,
                "agent_score": round(track.agent_score, 3),
                "team":       track.team,
                "x_norm":     round(track.last_cx, 4),
                "y_norm":     round(track.last_cy, 4),
                "has_spike":  track.has_spike,
            }
            positions.append(entry)

        if not positions:
            return []

        event: dict = {
            "type":      "POSITION_UPDATE",
            "t_ms":      t_ms,
            "round":     round_number,
            "positions": positions,
        }
        # Spike loose = on the floor or planted (not attached to any player)
        if spike_loose:
            event["spike_loose"] = {
                "x_norm": round(spike_loose[0], 4),
                "y_norm": round(spike_loose[1], 4),
            }

        # Independent spike track (ground / planted position across frames)
        if self._spike_track is not None:
            event["spike_track"] = {
                "x_norm":       round(self._spike_track.last_cx, 4),
                "y_norm":       round(self._spike_track.last_cy, 4),
                "state":        self._spike_track.state,
                "last_seen_ms": self._spike_track.last_seen_ms,
            }

        self._event_buffer.append(event)
        return [event]

    def notify_kill(
        self,
        t_ms: float,
        killer_name: str,
        victim_name: str,
        killer_team: str,
        victim_team: str,
    ) -> None:
        """
        Called when a KILL_EVENT is confirmed.  Resolves victim track by
        finding the track on victim_team that recently disappeared.
        Resolves killer when only one unresolved alive track remains.
        """
        kn = killer_name.lower()
        vn = victim_name.lower()

        # ── Victim ────────────────────────────────────────────────────────────
        if vn not in self._name_to_track:
            candidate = (
                self._closest_unresolved_death(victim_team, t_ms, 5000)
                or self._first_unresolved_alive(victim_team)
            )
            if candidate is not None:
                self._assign_name(candidate, vn)
                candidate.alive    = False
                candidate.death_ms = candidate.death_ms or t_ms
        else:
            tid = self._name_to_track.get(vn)
            if tid and tid in self._tracks:
                t = self._tracks[tid]
                t.alive    = False
                t.death_ms = t.death_ms or t_ms

        # ── Killer: name it if only one unresolved alive track on that team ───
        if kn not in self._name_to_track:
            unresolved = [
                t for t in self._tracks.values()
                if t.team == killer_team and t.player_name is None and t.alive
            ]
            if len(unresolved) == 1:
                self._assign_name(unresolved[0], kn)

    def resolve_round_identities(self, kill_events: List[dict]) -> None:
        """
        Post-round pass over kill events (sorted by time) to maximise the
        number of named tracks.  Each dict needs: t_ms, killer_name,
        victim_name, killer_team, victim_team  (teams as "left"/"right").
        """
        for ev in sorted(kill_events, key=lambda e: e.get("t_ms", 0)):
            self.notify_kill(
                t_ms=ev.get("t_ms", 0),
                killer_name=ev.get("killer_name", ""),
                victim_name=ev.get("victim_name", ""),
                killer_team=ev.get("killer_team", ""),
                victim_team=ev.get("victim_team", ""),
            )

    def get_last_position(
        self, player_name: str
    ) -> Optional[Tuple[float, float, float]]:
        """Return last known (x_norm, y_norm, t_ms) or None."""
        return self._last_positions.get(player_name.lower())

    def notify_spike_planted(self, t_ms: float) -> None:
        """
        Notify the tracker that the spike has been planted.
        Updates the spike track state from "ground" to "planted".
        Call this when a SPIKE_PLANT event is confirmed by the pipeline.
        """
        self._spike_planted = True
        if self._spike_track is not None:
            self._spike_track.state = "planted"

    def get_spike_track(self) -> Optional[SpikeTrack]:
        """Return the current independent spike track, or None if not seen."""
        return self._spike_track

    def flush_events(self) -> List[dict]:
        """Return and clear buffered POSITION_UPDATE events."""
        evs = list(self._event_buffer)
        self._event_buffer.clear()
        return evs

    # ── Private: detection ────────────────────────────────────────────────────

    def _find_players_by_template(
        self,
        minimap_bgr: np.ndarray,
        hsv: np.ndarray,
        h: int,
        w: int,
    ) -> Tuple[
        List[Tuple[float, float, float, float, Optional[str]]],
        List[Tuple[float, float, float, float, Optional[str]]],
    ]:
        """
        Slide all agent templates → NMS → ring-colour team assignment.
        Returns (left_dets, right_dets) where each det is
          (cx_norm, cy_norm, cx_px, cy_px, agent_name).
        """
        gray = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY)
        hits = self._agent_matcher.find_all(gray, self.AGENT_NCC_THRESHOLD)

        left_dets:  List[Tuple[float, float, float, float, Optional[str]]] = []
        right_dets: List[Tuple[float, float, float, float, Optional[str]]] = []
        for agent, cx_px, cy_px, _score in hits:
            team = self._sample_team(hsv, cx_px, cy_px, h, w)
            det  = (float(cx_px) / w, float(cy_px) / h,
                    float(cx_px),     float(cy_px), agent)
            if team == "left":
                left_dets.append(det)
            elif team == "right":
                right_dets.append(det)
            # else: no visible ring colour → discard (barrier / ability / noise)

        # Hits are already sorted best-first by AgentTemplateMatcher.find_all
        return left_dets[:self.MAX_PLAYERS], right_dets[:self.MAX_PLAYERS]

    def _sample_team(
        self,
        hsv: np.ndarray,
        cx: int,
        cy: int,
        h: int,
        w: int,
    ) -> Optional[str]:
        """
        Sample the annular ring around a template-matched icon centre.
        Returns "left", "right", or None if neither colour is dominant.
        """
        ri, ro = self.RING_SAMPLE_INNER, self.RING_SAMPLE_OUTER
        y1 = max(0, cy - ro);  y2 = min(h, cy + ro + 1)
        x1 = max(0, cx - ro);  x2 = min(w, cx + ro + 1)
        crop = hsv[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        ch, cw = crop.shape[:2]
        ys, xs = np.mgrid[0:ch, 0:cw]
        dist = np.hypot(xs - (cx - x1), ys - (cy - y1)).astype(np.float32)
        ring_mask = ((dist >= ri) & (dist <= ro)).astype(np.uint8) * 255

        teal_px = cv2.countNonZero(cv2.bitwise_and(
            cv2.inRange(crop, self._TEAL_LO, self._TEAL_HI), ring_mask
        ))
        red_px = cv2.countNonZero(cv2.bitwise_and(
            cv2.bitwise_or(
                cv2.inRange(crop, self._RED_LO1, self._RED_HI1),
                cv2.inRange(crop, self._RED_LO2, self._RED_HI2),
            ),
            ring_mask,
        ))

        mn = self.RING_MIN_PIXELS
        if teal_px >= mn and teal_px > red_px:
            return "left" if self.left_color == "teal" else "right"
        if red_px  >= mn and red_px  > teal_px:
            return "left" if self.left_color == "red"  else "right"
        return None

    def _find_blobs(
        self,
        hsv: np.ndarray,
        color: str,
        h: int,
        w: int,
    ) -> List[Tuple[float, float, float, float, float, float]]:
        """
        Colour-blob fallback (used when no agent templates are loaded).
        Returns (cx_norm, cy_norm, area_px, r_px, cx_px, cy_px) per blob.
        """
        if color == "teal":
            _teal_base    = cv2.inRange(hsv, self._TEAL_LO,  self._TEAL_HI)
            _teal_ext     = cv2.inRange(hsv, self._TEAL_LO2, self._TEAL_HI2)
            mask          = cv2.bitwise_or(_teal_base, _teal_ext)
            # The extended (blue-shifted) sub-mask after morphological close:
            # used per-contour to decide whether to allow a lower area floor.
            kern = self._KERNEL
            _teal_ext_closed = cv2.morphologyEx(_teal_ext, cv2.MORPH_CLOSE, kern)
        else:
            m1   = cv2.inRange(hsv, self._RED_LO1, self._RED_HI1)
            m2   = cv2.inRange(hsv, self._RED_LO2, self._RED_HI2)
            mask = cv2.bitwise_or(m1, m2)
            kern = self._KERNEL_RED
            _teal_ext_closed = None   # unused for red path

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        blobs = []
        # Precompute masks and constants used by both the main filter loop
        # and the large-blob split pass (they are frame-level, not per-contour).
        _cr     = 7    # half-size of portrait V-std patch
        _cr_ctr = 5    # half-size of centre-color fraction patch
        if color == "teal":
            # Use only the base teal range for the centre-colour fraction test so
            # that blue/purple face portraits from the extended mask (TEAL_LO2/HI2)
            # do not cause real agent rings to fail the ability-icon rejection filter.
            _ctr_color_mask = cv2.inRange(hsv, self._TEAL_LO, self._TEAL_HI)
        else:
            _ctr_color_mask = cv2.bitwise_or(
                cv2.inRange(hsv, self._CTR_RED_LO1, self._CTR_RED_HI1),
                cv2.inRange(hsv, self._CTR_RED_LO2, self._CTR_RED_HI2),
            )
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # For teal blobs that contain pixels from the extended blue-shifted
            # range (spike-occluded rings), allow a lower area floor.  Pure
            # original-teal blobs keep the strict 80 px² floor so that map
            # artifacts that are below that threshold are not promoted.
            if _teal_ext_closed is not None:
                bx_t, by_t, bw_t, bh_t = cv2.boundingRect(cnt)
                _has_ext = np.any(_teal_ext_closed[by_t:by_t+bh_t, bx_t:bx_t+bw_t])
                _min_area = self.MIN_AREA if _has_ext else 80
            else:
                _min_area = 80
            if not (_min_area <= area <= self.MAX_AREA):
                continue
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            r = float(r)
            if not (self.MIN_RADIUS <= r <= self.MAX_RADIUS):
                continue
            # Aspect-ratio filter: rejects elongated barriers / ability lines.
            bx, by, bw, bh = cv2.boundingRect(cnt)
            long_side  = max(bw, bh)
            short_side = max(1, min(bw, bh))
            if long_side / short_side > self.MAX_ASPECT_RATIO:
                continue

            # Solidity filter: rejects solid blobs (barriers, ability icons).
            # Ring arcs are hollow — area << convex hull area (solidity < 0.90).
            # Solid rectangles and circles have solidity ≈ 1.0.
            hull      = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0 and (area / hull_area) > self.MAX_BLOB_SOLIDITY:
                continue

            # Border margin: reject blobs too close to the minimap edge
            # (they are UI frame decorations, not player icons).
            bm = self.BORDER_MARGIN_PX
            if cx < bm or cx > w - bm or cy < bm or cy > h - bm:
                continue

            # ── Refine the ring centre with HoughCircles ──────────────────────
            # minEnclosingCircle gives the centre of the arc PIXELS, which drifts
            # toward the dense side when only part of the ring is detected.
            # HoughCircles accumulates votes along the arc and finds the geometric
            # ring centre even from partial (≥ ~90°) arcs.
            cx_ref, cy_ref = float(cx), float(cy)
            roi_r   = 20          # ROI half-size around MEC estimate
            rx1 = max(0, int(cx) - roi_r);  ry1 = max(0, int(cy) - roi_r)
            rx2 = min(w, int(cx) + roi_r);  ry2 = min(h, int(cy) + roi_r)
            roi_mask = mask[ry1:ry2, rx1:rx2]
            if roi_mask.size > 0:
                roi_blur = cv2.GaussianBlur(roi_mask, (3, 3), 1)
                hough = cv2.HoughCircles(
                    roi_blur, cv2.HOUGH_GRADIENT,
                    dp=1, minDist=10,
                    param1=50, param2=6,
                    minRadius=6, maxRadius=13,
                )
                if hough is not None:
                    hcx, hcy, _hr = hough[0][0]
                    rcx = rx1 + hcx;  rcy = ry1 + hcy
                    # Accept only if within 8 px of MEC estimate (sanity check)
                    if abs(rcx - cx) < 8 and abs(rcy - cy) < 8:
                        cx_ref, cy_ref = rcx, rcy

            # Portrait-centre texture filter: ability rings have a flat/empty
            # centre; real agent icons contain a face portrait (high V variance).
            _cyi = int(cy_ref); _cxi = int(cx_ref)
            centre_v = hsv[max(0, _cyi - _cr):_cyi + _cr + 1,
                           max(0, _cxi - _cr):_cxi + _cr + 1, 2]
            if centre_v.size >= 4 and float(np.std(centre_v)) < self.MIN_PORTRAIT_STDDEV:
                continue

            # Ability-icon colour-centre rejection (masks precomputed above).
            _ctr_patch = _ctr_color_mask[max(0, _cyi - _cr_ctr):_cyi + _cr_ctr + 1,
                                          max(0, _cxi - _cr_ctr):_cxi + _cr_ctr + 1]
            if (_ctr_patch.size >= 4
                    and float(np.count_nonzero(_ctr_patch)) / _ctr_patch.size
                    > self.MAX_CTR_TEAL_FRAC):
                continue

            # ── Large-blob split: recover individual circles from merged icons ──
            # Only attempt if the blob is spatially extended (major axis ≥ 32 px).
            # Stacked/heavily-overlapping icons produce compact, near-circular
            # blobs; genuinely side-by-side agents produce elongated ones.
            if area > self.SPLIT_AREA_THRESHOLD and max(bw, bh) >= 32:
                _split_pad = 8
                srx1 = max(0, bx - _split_pad);  sry1 = max(0, by - _split_pad)
                srx2 = min(w, bx + bw + _split_pad); sry2 = min(h, by + bh + _split_pad)
                split_blur = cv2.GaussianBlur(mask[sry1:sry2, srx1:srx2], (3, 3), 1)
                split_hough = cv2.HoughCircles(
                    split_blur, cv2.HOUGH_GRADIENT,
                    dp=1, minDist=14,
                    param1=50, param2=5,
                    minRadius=6, maxRadius=13,
                )
                if split_hough is not None and len(split_hough[0]) >= 2:
                    sub_blobs = []
                    for shcx, shcy, shr in split_hough[0]:
                        scx = float(srx1 + shcx)
                        scy = float(sry1 + shcy)
                        if (scx < self.BORDER_MARGIN_PX or scx > w - self.BORDER_MARGIN_PX
                                or scy < self.BORDER_MARGIN_PX or scy > h - self.BORDER_MARGIN_PX):
                            continue
                        _scxi, _scyi = int(scx), int(scy)
                        # Centre-color check
                        _sp = _ctr_color_mask[
                            max(0, _scyi - _cr_ctr):_scyi + _cr_ctr + 1,
                            max(0, _scxi - _cr_ctr):_scxi + _cr_ctr + 1]
                        if (_sp.size >= 4 and
                                float(np.count_nonzero(_sp)) / _sp.size > self.MAX_CTR_TEAL_FRAC):
                            continue
                        # Portrait texture check
                        _sv = hsv[max(0, _scyi - _cr):_scyi + _cr + 1,
                                   max(0, _scxi - _cr):_scxi + _cr + 1, 2]
                        if _sv.size >= 4 and float(np.std(_sv)) < self.MIN_PORTRAIT_STDDEV:
                            continue
                        sub_blobs.append((
                            scx / w, scy / h,
                            shr * shr * 3.14159, shr,
                            scx, scy,
                        ))
                    if len(sub_blobs) >= 2:
                        blobs.extend(sub_blobs)
                        continue  # merged blob replaced by its sub-circles

            blobs.append((
                cx_ref / w,   # cx_norm
                cy_ref / h,   # cy_norm
                area,          # area_px
                r,             # r_px
                cx_ref,        # cx_px
                cy_ref,        # cy_px
            ))

        # ── Near-duplicate suppression ────────────────────────────────────────
        # Two same-team blobs < 12 px apart are physically impossible (icon
        # diameter ≈ 22 px).  Keep the larger-area blob; drop the smaller one.
        # This eliminates face-pixel artefacts that sit very close to a
        # properly-detected ring blob (e.g. clove face in the red mask).
        if len(blobs) > 1:
            _DEDUP_R2 = 12.0 ** 2   # squared pixel radius
            kept: List[Tuple] = []
            # Sort by area descending so the dominant blob is always kept.
            for b in sorted(blobs, key=lambda x: x[2], reverse=True):
                _bx, _by = b[4], b[5]
                if any((_bx - k[4]) ** 2 + (_by - k[5]) ** 2 < _DEDUP_R2
                       for k in kept):
                    continue
                kept.append(b)
            blobs = kept

        return blobs

    def _find_spike_blobs(
        self, hsv: np.ndarray, h: int, w: int
    ) -> List[Tuple[float, float, float]]:
        """
        Returns (cx_norm, cy_norm, area_px) for yellow spike blobs.
        Covers carrier indicator, spike on floor, and spike planted.
        """
        mask = cv2.inRange(hsv, self._SPIKE_LO, self._SPIKE_HI)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        blobs = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.SPIKE_MIN_AREA <= area <= self.SPIKE_MAX_AREA):
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            blobs.append((
                float(M["m10"] / M["m00"]) / w,
                float(M["m01"] / M["m00"]) / h,
                area,
            ))
        return blobs

    def _assign_spike(
        self,
        spike_blobs: List[Tuple[float, float, float]],
        t_ms: float = 0.0,
    ) -> Optional[Tuple[float, float]]:
        """
        For each yellow blob:
          - If within SPIKE_CARRIER_MAX_DIST of an alive player → has_spike=True.
          - Otherwise → record as loose spike position and update _spike_track.
        Returns the first loose spike (x_norm, y_norm) found, or None.
        """
        # Clear flags from previous frame
        for track in self._tracks.values():
            track.has_spike = False

        loose: Optional[Tuple[float, float]] = None

        for cx_n, cy_n, _area in spike_blobs:
            best_track = None
            best_dist  = self.SPIKE_CARRIER_MAX_DIST
            for track in self._tracks.values():
                if not track.alive:
                    continue
                d = math.hypot(cx_n - track.last_cx, cy_n - track.last_cy)
                if d < best_dist:
                    best_dist  = d
                    best_track = track

            if best_track is not None:
                best_track.has_spike = True
                # Spike is now carried — clear the independent ground track so
                # the previous drop position is not shown alongside the carrier.
                # Keep the track once planted (spike can't be un-planted).
                if not self._spike_planted:
                    self._spike_track = None
            else:
                loose = (cx_n, cy_n)
                # Update the independent spike track (ground / planted)
                _state = "planted" if self._spike_planted else "ground"
                if self._spike_track is None:
                    self._spike_track = SpikeTrack(cx_n, cy_n, t_ms, _state)
                else:
                    self._spike_track.state = _state
                    self._spike_track.update(cx_n, cy_n, t_ms)

        return loose

    # ── Private: tracking ─────────────────────────────────────────────────────

    # How much better a new NCC score must be over the stored score to
    # trigger an agent re-identification on an existing track.
    _AGENT_UPDATE_MARGIN = 0.06

    def _update_team_tracks(
        self,
        dets: List[Tuple],
        team: str,
        t_ms: float,
    ) -> None:
        """
        Nearest-neighbour assignment for one team.
        dets = [(cx_norm, cy_norm, cx_px, cy_px, agent_name_or_None, ncc_score,
                 all_scores_sorted)]
        Agent name is updated when the new NCC score beats the stored score
        by _AGENT_UPDATE_MARGIN.  Per-team uniqueness is enforced: if two
        blobs would both claim the same agent, the lower-scoring blob falls
        back to its next-best agent that is not yet claimed.
        """
        team_tracks = {
            tid: t
            for tid, t in self._tracks.items()
            if t.team == team and t.alive
        }
        used_tids: set = set()

        # Sort detections by NCC score descending so high-confidence blobs
        # claim their agent first.
        sorted_dets = sorted(dets, key=lambda d: d[5], reverse=True)

        # claimed_agents: agents already locked by existing tracks this frame.
        # We build this as we assign detections to existing tracks, so the
        # set grows through the loop and later new-track creations respect it.
        claimed_agents: set = {
            t.agent_name
            for t in team_tracks.values()
            if t.agent_name is not None
        }

        for cx_n, cy_n, _cx_px, _cy_px, agent_name, agent_score, all_scores in sorted_dets:
            # Enforce per-team uniqueness + known-roster constraint.
            # If the top agent is already claimed OR not in this team's roster,
            # fall back to the next-best agent that passes both filters.
            known_set = (
                self._known_left_agents  if team == "left"
                else self._known_right_agents
            )
            final_agent  = agent_name
            final_score  = agent_score
            # Reject if claimed or off-roster
            if final_agent is not None and (
                final_agent in claimed_agents
                or (known_set is not None and final_agent not in known_set)
            ):
                final_agent = None
                final_score = 0.0
            if final_agent is None and all_scores:
                for a, s in all_scores:
                    if s < self.FACE_CROP_NCC_THRESHOLD:
                        break
                    if a not in claimed_agents and (
                        known_set is None or a in known_set
                    ):
                        final_agent = a
                        final_score = s
                        break

            if final_agent:
                claimed_agents.add(final_agent)

            best_tid  = None
            best_dist = self.MAX_ASSIGN_DIST
            for tid, track in team_tracks.items():
                if tid in used_tids:
                    continue
                d = math.hypot(cx_n - track.last_cx, cy_n - track.last_cy)
                if d < best_dist:
                    best_dist = d
                    best_tid  = tid

            if best_tid is not None:
                t = self._tracks[best_tid]
                t.last_cx       = cx_n
                t.last_cy       = cy_n
                t.last_seen_ms  = t_ms
                t.missing_count = 0
                t.positions.append((t_ms, cx_n, cy_n))
                # Update agent name if this frame has a meaningfully better score
                if final_agent and (
                    t.agent_name is None
                    or final_score > t.agent_score + self._AGENT_UPDATE_MARGIN
                ):
                    t.agent_name             = final_agent
                    t.agent_score            = final_score
                    t.unidentified_frames    = 0
                    if t.player_name is None:
                        _team_map = self._left_agent_to_player if t.team == "left" else self._right_agent_to_player
                        p = _team_map.get(final_agent) or self._agent_to_player.get(final_agent)
                        if p:
                            self._assign_name(t, p)
                elif t.agent_name is None:
                    # No agent matched this frame; count consecutive unidentified frames.
                    # Expire tracks that consistently fail NCC (e.g. ability markers).
                    t.unidentified_frames += 1
                    if t.unidentified_frames >= self.MAX_NO_AGENT_FRAMES:
                        t.alive    = False
                        t.death_ms = t_ms
                used_tids.add(best_tid)
            else:
                alive_n = sum(
                    1 for tt in self._tracks.values()
                    if tt.team == team and tt.alive
                )
                if alive_n < self.MAX_PLAYERS:
                    idx = self._next_idx[team]
                    self._next_idx[team] += 1
                    tid = f"{team}_{idx}"
                    new_t = BlobTrack(tid, team, cx_n, cy_n, t_ms, final_agent)
                    new_t.agent_score = final_score
                    if final_agent:
                        _team_map = self._left_agent_to_player if team == "left" else self._right_agent_to_player
                        p = _team_map.get(final_agent) or self._agent_to_player.get(final_agent)
                        if p:
                            self._assign_name(new_t, p)
                    self._tracks[tid] = new_t
                    used_tids.add(tid)

        for tid, track in team_tracks.items():
            if tid not in used_tids:
                track.missing_count += 1
                if track.missing_count >= self.DEAD_AFTER_N:
                    track.alive    = False
                    track.death_ms = t_ms

    # ── Private: identity ─────────────────────────────────────────────────────

    def _deduplicate_agents_across_teams(self) -> None:
        """
        Position-based ghost suppression: if an *unidentified* track
        (agent_name is None) sits within ~17 px of any alive track on the
        *opposite* team, it is almost certainly a false positive caused by the
        opponent agent's face-portrait skin tones leaking into the detector
        (e.g. Breach's warm-orange face pixels triggering the red-ring mask).
        Mark such ghost tracks dead so they do not pollute the output.

        Note: per-team agent uniqueness is already enforced inside
        _update_team_tracks via claimed_agents, so no cross-team name dedup
        is needed here (both teams can legitimately play the same agent).
        """
        # ── Position-based ghost suppression ──────────────────────────────────
        # Threshold is in *normalised* coordinate space (coords are in [0, 1]).
        # 12 px on a ~450 px wide minimap  →  12/450 ≈ 0.0267  →  squared ≈ 0.00071.
        # Use 0.0012 for a comfortable ≈17 px margin.
        _GHOST_R2 = 0.0012
        alive_tracks = [t for t in self._tracks.values() if t.alive]
        for track in alive_tracks:
            if track.agent_name is not None:
                continue  # already identified — keep it
            opposite = "right" if track.team == "left" else "left"
            for other in alive_tracks:
                if other.team != opposite:
                    continue
                dist2 = (track.last_cx - other.last_cx) ** 2 + \
                        (track.last_cy - other.last_cy) ** 2
                if dist2 < _GHOST_R2:
                    # Unidentified track shadows an opponent — ghost, kill it.
                    track.alive = False
                    break

    def _assign_name(self, track: BlobTrack, player_name: str) -> None:
        """
        Assign a player name to a track and update cross-round mappings.
        No-op if the track already carries a different name.
        """
        if track.player_name and track.player_name != player_name:
            return
        track.player_name = player_name
        self._name_to_track[player_name] = track.track_id
        self._last_positions[player_name] = (
            track.last_cx, track.last_cy, track.last_seen_ms
        )
        if track.agent_name:
            self._agent_to_player[track.agent_name] = player_name
            self._player_to_agent[player_name]       = track.agent_name

    def _closest_unresolved_death(
        self, team: str, t_ms: float, window_ms: float
    ) -> Optional[BlobTrack]:
        best: Optional[BlobTrack] = None
        best_dt = window_ms
        for track in self._tracks.values():
            if track.team != team or track.player_name is not None:
                continue
            if track.death_ms is None:
                continue
            dt = abs(track.death_ms - t_ms)
            if dt < best_dt:
                best_dt = dt
                best    = track
        return best

    def _first_unresolved_alive(self, team: str) -> Optional[BlobTrack]:
        for track in self._tracks.values():
            if track.team == team and track.player_name is None and track.alive:
                return track
        return None



