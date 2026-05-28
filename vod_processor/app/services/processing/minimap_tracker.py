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
        sr     = search_r
        cx_i   = round(cx_px)
        cy_i   = round(cy_px)
        best_name:  Optional[str] = None
        best_score: float         = threshold - 1e-6
        best_dx: int = 0
        best_dy: int = 0
        best_all: Dict[str, float] = {a: -1.0 for a in self._templates}

        for dy in range(-sr, sr + 1):
            for dx in range(-sr, sr + 1):
                patch = self._masked_patch(minimap_bgr, cx_i + dx, cy_i + dy)
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
    FACE_CROP_NCC_THRESHOLD = 0.22  # NCC required to name an agent from its blob

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
    ) -> None:
        self.left_players  = [p.lower() for p in left_players]
        self.right_players = [p.lower() for p in right_players]
        self.left_color    = left_color
        self.right_color   = right_color

        # Agent template matcher (optional)
        self._agent_matcher: Optional[AgentTemplateMatcher] = None
        if agents_dir:
            m = AgentTemplateMatcher(agents_dir)
            if m.ready:
                self._agent_matcher = m

        # Cross-round agent ↔ player mappings (persist across reset_round)
        self._agent_to_player: Dict[str, str] = {
            a.lower(): p.lower() for a, p in (agent_map or {}).items()
        }
        self._player_to_agent: Dict[str, str] = {
            v: k for k, v in self._agent_to_player.items()
        }

        # Per-round state (cleared by reset_round)
        self._tracks: Dict[str, BlobTrack] = {}
        self._next_idx: Dict[str, int] = {"left": 0, "right": 0}
        self._name_to_track: Dict[str, str] = {}
        self._last_positions: Dict[str, Tuple[float, float, float]] = {}
        self._round_number = 0
        self._event_buffer: List[dict] = []

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
        # Save crops from the full pre-cap blob list so we don't miss any
        # potential player icons that were pushed out by the cap.
        if self._crop_dump_dir and (self._frame_count - 1) % self._crop_save_stride == 0:
            sep = self._crop_min_sep_px
            for tag, blobs_all in (("L", left_blobs_all), ("R", right_blobs_all)):
                for idx, (_, _, _area, _r, cx_px, cy_px) in enumerate(blobs_all):
                    overlapping = any(
                        (ox - cx_px) ** 2 + (oy - cy_px) ** 2 < sep * sep
                        for ox, oy in all_blob_px
                        if not (ox == cx_px and oy == cy_px)
                    )
                    if not overlapping:
                        self._save_blob_crop(
                            minimap_bgr, cx_px, cy_px, f"{tag}{idx}"
                        )

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
                else:
                    all_scores = []
                    agent, score = None, 0.0
                dets.append((cx_n, cy_n, cx_px, cy_px, agent, score, all_scores))
            return dets

        left_dets  = _dets_from_blobs(left_blobs,  "L")
        right_dets = _dets_from_blobs(right_blobs, "R")

        self._update_team_tracks(left_dets,  "left",  t_ms)
        self._update_team_tracks(right_dets, "right", t_ms)
        # Revoke the lower-scoring copy when the same agent is claimed on both
        # teams (impossible in Valorant — each agent is unique per match).
        self._deduplicate_agents_across_teams()

        # ── Spike blobs (always yellow) ───────────────────────────────────────
        spike_blobs = self._find_spike_blobs(hsv, h, w)
        spike_loose = self._assign_spike(spike_blobs)   # (x_n, y_n) or None

        # ── Update last-known positions for named tracks ──────────────────────
        for track in self._tracks.values():
            if track.alive and track.player_name:
                self._last_positions[track.player_name] = (
                    track.last_cx, track.last_cy, t_ms
                )

        # ── Build event ───────────────────────────────────────────────────────
        positions = []
        for track in self._tracks.values():
            entry: dict = {
                "track_id":   track.track_id,
                "player":     track.player_name or f"unknown_{track.track_id}",
                "agent":      track.agent_name,
                "agent_score": round(track.agent_score, 3),
                "team":       track.team,
                "x_norm":     round(track.last_cx, 4),
                "y_norm":     round(track.last_cy, 4),
                "alive":      track.alive,
                "has_spike":  track.has_spike,
            }
            if not track.alive and track.death_ms is not None:
                entry["death_ms"] = track.death_ms
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
    ) -> Optional[Tuple[float, float]]:
        """
        For each yellow blob:
          - If within SPIKE_CARRIER_MAX_DIST of an alive player → has_spike=True.
          - Otherwise → record as loose spike position.
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
            else:
                loose = (cx_n, cy_n)

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
            # Enforce per-team uniqueness: if the top agent is already claimed
            # this frame, fall back to the next-best unique agent.
            final_agent  = agent_name
            final_score  = agent_score
            if all_scores and agent_name in claimed_agents:
                final_agent = None
                final_score = 0.0
                for a, s in all_scores:
                    if s < self.FACE_CROP_NCC_THRESHOLD:
                        break
                    if a not in claimed_agents:
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
                        p = self._agent_to_player.get(final_agent)
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
                        p = self._agent_to_player.get(final_agent)
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
        Revoke the lower-scoring assignment when the same agent name appears
        on both teams.  In Valorant each agent is unique per match so a
        cross-team duplicate is always a misidentification.
        """
        seen: Dict[str, "BlobTrack"] = {}
        for track in self._tracks.values():
            if not track.alive or track.agent_name is None:
                continue
            agent = track.agent_name
            if agent in seen:
                prev = seen[agent]
                if track.agent_score > prev.agent_score:
                    prev.agent_name  = None
                    prev.agent_score = 0.0
                    seen[agent] = track
                else:
                    track.agent_name  = None
                    track.agent_score = 0.0
            else:
                seen[agent] = track

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



