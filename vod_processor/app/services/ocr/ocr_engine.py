"""
Enhanced OCR Engine with GPU acceleration and multi-pass processing.

Provides a unified interface for OCR with multiple preprocessing strategies
to maximize text detection accuracy for gaming killfeed text.

Supports:
- PaddleOCR (primary) - Fast and accurate, good balance of speed/quality
- Surya (fallback) - State-of-art transformer OCR for scene text  
- EasyOCR (fallback) - Reliable GPU-accelerated OCR
"""

import os
import re
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from difflib import SequenceMatcher

# Allow callers to explicitly disable OCR backend initialization (useful for
# lightweight crop-only runs). Set DISABLE_OCR=true in the environment to
# skip any Paddle/Surya/EasyOCR imports and initialization.
if os.environ.get('DISABLE_OCR', 'false').lower() == 'true':
    print("[OCR Engine] DISABLE_OCR=true, skipping OCR backend initialization", flush=True)
else:
    # Initialize PaddlePaddle GPU device early to ensure cuDNN is loaded
    # This must happen BEFORE importing PaddleOCR
    try:
        import paddle
        if os.environ.get('USE_GPU', 'false').lower() == 'true':
            paddle.device.set_device('gpu:0')
            print(f"[OCR Engine] Pre-initialized PaddlePaddle on gpu:0", flush=True)
    except Exception as e:
        print(f"[OCR Engine] Could not pre-initialize PaddlePaddle GPU: {e}", flush=True)


@dataclass
class OCRResult:
    """Represents a single OCR detection."""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    preprocessing: str = ""  # Which preprocessing produced this result


@dataclass 
class MultiPassResult:
    """Combined result from multiple OCR passes."""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    vote_count: int = 1
    sources: List[str] = field(default_factory=list)


class OCREngine:
    """
    Advanced OCR engine with multi-pass processing for maximum accuracy.
    
    Features:
    - PaddleOCR as primary backend (fast + accurate)
    - Surya as fallback (state-of-art for scene text)
    - EasyOCR as last fallback
    - Multiple preprocessing strategies run in parallel
    - Results voting/consensus for higher accuracy
    - GPU acceleration
    - Optimized for gaming text (killfeed, HUD elements)
    """
    
    def __init__(self, use_gpu: bool = True, prefer_surya: bool = True, prefer_paddle: bool = True):
        """Initialize OCR engine. PaddleOCR is preferred for speed with GPU."""
        self.use_gpu = use_gpu
        self.prefer_surya = prefer_surya
        self.prefer_paddle = prefer_paddle
        self._surya_foundation = None
        self._surya_predictor = None
        self._surya_det_predictor = None
        self._easyocr_reader = None
        self._paddleocr_reader = None
        self._backend = None
        self._initialized = False
        
    def _lazy_init(self):
        """Lazily initialize the OCR backend."""
        if self._initialized:
            return
            
        self._initialized = True
        
        # If OCR has been globally disabled via environment, skip init.
        if os.environ.get('DISABLE_OCR', 'false').lower() == 'true':
            self._backend = None
            print("[OCR Engine] Skipping backend initialization (DISABLE_OCR=true)", flush=True)
            return

        # Try PaddleOCR first (fast + accurate balance)
        # GPU mode with CUDA/cuDNN library symlinks configured in Dockerfile
        if self.prefer_paddle:
            try:
                from paddleocr import PaddleOCR
                import logging
                # Suppress PaddleOCR logging
                logging.getLogger('ppocr').setLevel(logging.ERROR)
                
                # Use GPU mode - CUDA/cuDNN symlinks configured in container
                use_gpu_paddle = self.use_gpu and os.environ.get('USE_GPU', 'false').lower() == 'true'
                
                # Initialize PaddleOCR v2.x
                self._paddleocr_reader = PaddleOCR(
                    use_angle_cls=False,  # Disable angle classification for speed
                    lang='en',
                    use_gpu=use_gpu_paddle,
                    show_log=False,
                    det_db_thresh=0.3,  # Detection threshold
                    rec_batch_num=6,  # Batch size for recognition
                )
                self._backend = "paddleocr"
                gpu_status = "GPU" if use_gpu_paddle else "CPU"
                print(f"[OCR Engine] Initialized PaddleOCR ({gpu_status} mode)", flush=True)
                return
            except Exception as e:
                print(f"[OCR Engine] PaddleOCR unavailable: {e}", flush=True)
        
        # Fallback: Try Surya (state-of-art for scene text)
        if self.prefer_surya:
            try:
                from surya.recognition import FoundationPredictor, RecognitionPredictor
                from surya.detection import DetectionPredictor
                import torch
                
                # Check GPU availability
                device = "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"
                
                # Initialize Surya predictors (0.17+ API requires FoundationPredictor)
                foundation = FoundationPredictor(device=device)
                self._surya_foundation = foundation
                self._surya_det_predictor = DetectionPredictor()
                self._surya_predictor = RecognitionPredictor(foundation)
                self._backend = "surya"
                
                if device == "cuda":
                    device_name = torch.cuda.get_device_name(0)
                    print(f"[OCR Engine] Initialized Surya with GPU ({device_name})", flush=True)
                else:
                    print(f"[OCR Engine] Initialized Surya (CPU mode)", flush=True)
                return
            except Exception as e:
                print(f"[OCR Engine] Surya unavailable: {e}", flush=True)
        
        # Fallback: EasyOCR (reliable with good GPU support)
        try:
            import easyocr
            import torch
            
            # Check if GPU is actually available
            gpu_available = torch.cuda.is_available()
            use_gpu = self.use_gpu and gpu_available
            
            self._easyocr_reader = easyocr.Reader(
                ['en'], 
                gpu=use_gpu, 
                verbose=False
            )
            self._backend = "easyocr"
            
            if use_gpu:
                device_name = torch.cuda.get_device_name(0)
                print(f"[OCR Engine] Initialized EasyOCR with GPU ({device_name})", flush=True)
            else:
                print(f"[OCR Engine] Initialized EasyOCR (CPU mode)", flush=True)
        except Exception as e:
            print(f"[OCR Engine] EasyOCR unavailable: {e}", flush=True)
            self._backend = None
    
    @property
    def backend(self) -> Optional[str]:
        """Return the active backend name."""
        self._lazy_init()
        return self._backend
    
    # ========================================
    # Preprocessing Methods
    # ========================================
    
    def _preprocess_original(self, image: np.ndarray) -> np.ndarray:
        """Return original image scaled up (2x for better OCR)."""
        return cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    def _preprocess_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE (2x scale + sharpening for gaming fonts)."""
        scaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        # Apply sharpening kernel first to enhance text edges
        sharpen_kernel = np.array([[-0.5, -0.5, -0.5],
                                   [-0.5,  5.0, -0.5],
                                   [-0.5, -0.5, -0.5]])
        sharpened = cv2.filter2D(scaled, -1, sharpen_kernel)
        # Then apply CLAHE for contrast
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _preprocess_sharpen(self, image: np.ndarray) -> np.ndarray:
        """Sharpen image for clearer text edges (2x scale for speed)."""
        scaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(scaled, -1, kernel)
    
    def _preprocess_denoise(self, image: np.ndarray) -> np.ndarray:
        """Denoise image to reduce artifacts (2x scale for speed)."""
        scaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        return cv2.fastNlMeansDenoisingColored(scaled, None, 10, 10, 7, 21)
    
    def _preprocess_threshold_white(self, image: np.ndarray) -> np.ndarray:
        """Binary threshold for white text on dark background (2x scale for speed)."""
        scaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    def _preprocess_adaptive(self, image: np.ndarray) -> np.ndarray:
        """Adaptive thresholding for varying lighting (2x scale for speed)."""
        scaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    def _preprocess_high_contrast(self, image: np.ndarray) -> np.ndarray:
        """Aggressive contrast enhancement (2x scale for speed)."""
        scaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        # Convert to LAB and boost L channel
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # Normalize L channel
        l = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
        # Apply strong CLAHE
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _preprocess_morphology(self, image: np.ndarray) -> np.ndarray:
        """Use morphological operations to clean up text (2x scale for speed)."""
        scaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        # Dilate to connect text components
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(gray, kernel, iterations=1)
        # Threshold
        _, thresh = cv2.threshold(dilated, 150, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    def _preprocess_saturation(self, image: np.ndarray) -> np.ndarray:
        """Isolate white text from coloured backgrounds using HSV saturation+value.
        
        White text has low saturation and high value, while teal/orange
        backgrounds have high saturation.  This produces a clean binary mask
        that works much better than greyscale thresholding for same-colour
        (self-kill) killfeed rows.
        """
        scaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # White text: S < 80 AND V > 170
        low_sat = cv2.threshold(s, 80, 255, cv2.THRESH_BINARY_INV)[1]
        high_val = cv2.threshold(v, 170, 255, cv2.THRESH_BINARY)[1]
        white_mask = cv2.bitwise_and(low_sat, high_val)
        # Light morphology to connect broken strokes
        kernel = np.ones((2, 2), np.uint8)
        white_mask = cv2.dilate(white_mask, kernel, iterations=1)
        return cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)

    # ========================================
    # Core OCR Methods
    # ========================================
    
    def read_text(self, image: np.ndarray, 
                  min_confidence: float = 0.3) -> List[OCRResult]:
        """
        Read text using single-pass OCR with default preprocessing.
        
        For better accuracy, use read_text_multipass().
        """
        self._lazy_init()
        if self._paddleocr_reader is None and self._surya_predictor is None and self._easyocr_reader is None:
            return []
        
        preprocessed = self._preprocess_contrast(image)
        return self._run_ocr(preprocessed, min_confidence, "contrast")
    
    def read_text_multipass(self, image: np.ndarray,
                            min_confidence: float = 0.2,
                            strategies: List[str] = None) -> List[MultiPassResult]:
        """
        Run OCR with multiple preprocessing strategies and combine results.
        
        This is the recommended method for maximum accuracy.
        
        Args:
            image: BGR image
            min_confidence: Minimum confidence for individual results
            strategies: List of preprocessing strategies to use.
                       Options: 'original', 'contrast', 'sharpen', 'denoise',
                               'threshold', 'adaptive', 'high_contrast',
                               'morphology', 'saturation'
                       Default: ['original', 'contrast', 'sharpen', 'high_contrast']
        
        Returns:
            List of MultiPassResult with voting/consensus information
        """
        self._lazy_init()
        if self._paddleocr_reader is None and self._surya_predictor is None and self._easyocr_reader is None:
            return []
        
        if strategies is None:
            # Single strategy for speed (contrast provides best results)
            strategies = ['contrast']
        
        # Map strategy names to preprocessing functions
        strategy_map = {
            'original': self._preprocess_original,
            'contrast': self._preprocess_contrast,
            'sharpen': self._preprocess_sharpen,
            'denoise': self._preprocess_denoise,
            'threshold': self._preprocess_threshold_white,
            'adaptive': self._preprocess_adaptive,
            'high_contrast': self._preprocess_high_contrast,
            'morphology': self._preprocess_morphology,
            'saturation': self._preprocess_saturation,
        }
        
        # Collect all results from all strategies
        all_results: List[OCRResult] = []
        
        for strategy_name in strategies:
            if strategy_name not in strategy_map:
                continue
            
            preprocess_fn = strategy_map[strategy_name]
            try:
                preprocessed = preprocess_fn(image)
                results = self._run_ocr(preprocessed, min_confidence, strategy_name)
                all_results.extend(results)
            except Exception as e:
                print(f"[OCR Engine] Strategy '{strategy_name}' failed: {e}", flush=True)
        
        # Combine results using voting
        return self._combine_results(all_results)
    
    def _is_garbage_text(self, text: str) -> bool:
        """
        Filter out garbage OCR output that Surya hallucinates.
        
        Returns True if the text should be rejected.
        """
        if not text or len(text.strip()) < 2:
            return True
        
        text = text.strip()
        text_lower = text.lower()
        
        # Filter 1: Pure numbers, symbols, or very short text
        if text.isdigit():
            return True
        if len(text) < 3:
            return True
        if not any(c.isalpha() for c in text):
            return True
        
        # Filter 2: LaTeX/HTML/markup patterns (Surya hallucinates these)
        garbage_patterns = [
            r'<math>',  # LaTeX math
            r'</math>',
            r'<b>',     # HTML bold
            r'</b>',
            r'<u>',     # HTML underline
            r'</u>',
            r'\\overline',  # LaTeX
            r'\\phantom',
            r'\\boldsymbol',
            r'\\cal',
            r'\\it',
            r'\$',      # Dollar signs (not player names)
        ]
        for pattern in garbage_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Filter 3: Repetitive text ("the state of the state of the state...")
        # Check if text has high repetition of phrases
        words = text_lower.split()
        if len(words) >= 6:
            # Count word frequencies
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            # If any word appears more than 40% of the time, it's repetitive
            max_freq = max(word_counts.values()) if word_counts else 0
            if max_freq > len(words) * 0.4:
                return True
        
        # Filter 4: Common garbage patterns from graphical noise
        garbage_strings = [
            '----', '____', '....', '. . .', '- - -',
            '* * *', '***', '===', '---',
            'the state of', 'the second', 'the same of',
            'the party of', 'the property of', 'the person',
            'column 2', 'column two', 'in column',
            'a contractor', 'a real property', 'a security',
            'the reserve', 'the residence', 'control of the',
            'name of persons', 'named in column',
        ]
        for garbage in garbage_strings:
            if garbage in text_lower:
                return True
        
        # Filter 5: Text that's mostly the same character repeated
        if len(text) > 4:
            char_counts = {}
            for c in text_lower:
                if c.isalnum():
                    char_counts[c] = char_counts.get(c, 0) + 1
            if char_counts:
                most_common = max(char_counts.values())
                total_alphanum = sum(char_counts.values())
                if most_common > total_alphanum * 0.6:
                    return True
        
        # Filter 6: Too long (player names with team prefix are typically < 20 chars)
        if len(text) > 30:
            return True
        
        # Filter 7: Unicode garbage (Surya sometimes outputs weird chars)
        if any(ord(c) > 127 for c in text if c not in '\u2019\u0027'):  # Allow apostrophe
            # Check if it's mostly ASCII - if less than 50% ASCII letters, reject
            ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
            if ascii_letters < len(text) * 0.5:
                return True
        
        return False
    
    def _run_ocr(self, image: np.ndarray, min_confidence: float, 
                 preprocessing: str) -> List[OCRResult]:
        """Run OCR on preprocessed image using available backend."""
        results = []
        
        try:
            if self._paddleocr_reader is not None:
                # PaddleOCR - convert BGR to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # PaddleOCR returns: [[[box], (text, confidence)], ...]
                # where box is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                ocr_results = self._paddleocr_reader.ocr(rgb_image, cls=False)
                
                # Handle None results or empty results
                if ocr_results and ocr_results[0]:
                    for line in ocr_results[0]:
                        box = line[0]  # [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                        text, confidence = line[1]
                        
                        if confidence < min_confidence:
                            continue
                        
                        # Filter garbage OCR output
                        if self._is_garbage_text(text):
                            continue
                        
                        # Convert polygon to bounding box (x, y, w, h)
                        xs = [p[0] for p in box]
                        ys = [p[1] for p in box]
                        x, y = int(min(xs)), int(min(ys))
                        w, h = int(max(xs) - x), int(max(ys) - y)
                        
                        results.append(OCRResult(
                            text=text.strip(),
                            confidence=confidence,
                            bbox=(x, y, w, h),
                            preprocessing=preprocessing
                        ))
            
            elif self._surya_predictor is not None:
                # Surya OCR - convert BGR to RGB for PIL
                from PIL import Image
                
                # Convert OpenCV BGR to RGB PIL Image
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                
                # Surya 0.17+ API: pass det_predictor parameter for combined detection+recognition
                # The recognition predictor will internally run detection first
                rec_results = self._surya_predictor(
                    [pil_image],
                    det_predictor=self._surya_det_predictor
                )
                
                if rec_results and len(rec_results) > 0:
                    rec_result = rec_results[0]
                    
                    # Process each text line
                    for text_line in rec_result.text_lines:
                        text = text_line.text
                        confidence = text_line.confidence
                        bbox = text_line.bbox  # [x1, y1, x2, y2]
                        
                        if confidence < min_confidence:
                            continue
                        
                        # Filter garbage OCR output
                        if self._is_garbage_text(text):
                            continue
                        
                        # Convert to x, y, w, h format
                        x, y = int(bbox[0]), int(bbox[1])
                        w, h = int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
                        
                        results.append(OCRResult(
                            text=text.strip(),
                            confidence=confidence,
                            bbox=(x, y, w, h),
                            preprocessing=preprocessing
                        ))
            
            elif self._easyocr_reader is not None:
                ocr_results = self._easyocr_reader.readtext(image)
                
                for bbox, text, confidence in ocr_results:
                    if confidence < min_confidence:
                        continue
                    
                    # Filter garbage OCR output
                    if self._is_garbage_text(text):
                        continue
                    
                    # Convert polygon to bounding box
                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]
                    x, y = int(min(xs)), int(min(ys))
                    w, h = int(max(xs) - x), int(max(ys) - y)
                    
                    results.append(OCRResult(
                        text=text.strip(),
                        confidence=confidence,
                        bbox=(x, y, w, h),
                        preprocessing=preprocessing
                    ))
                
        except Exception as e:
            print(f"[OCR Engine] OCR error ({preprocessing}): {e}", flush=True)
        
        return results
    
    def _combine_results(self, all_results: List[OCRResult]) -> List[MultiPassResult]:
        """
        Combine results from multiple passes using voting and text similarity.
        
        Results that appear in multiple passes get higher confidence.
        Similar texts are grouped together based on position and text similarity.
        """
        if not all_results:
            return []
        
        # Cluster results by position and text similarity
        clusters: List[List[OCRResult]] = []
        
        for result in all_results:
            matched_cluster = None
            
            # Find a cluster that this result belongs to
            for cluster in clusters:
                # Check if position is close AND text is similar.
                # IMPORTANT: always require position proximity so that identical
                # text at different positions (e.g. self-kills in Valorant where
                # the same player name appears on both sides) stays as separate
                # clusters.
                for existing in cluster:
                    x_diff = abs(result.bbox[0] - existing.bbox[0])
                    text_sim = self._text_similarity(result.text, existing.text)
                    
                    # Match if: close position AND text is similar
                    # (at 2x scale, same name across strategies varies by <50px;
                    #  different names on same row are 100-400px apart)
                    if x_diff < 100 and text_sim > 0.5:
                        matched_cluster = cluster
                        break
                
                if matched_cluster:
                    break
            
            if matched_cluster:
                matched_cluster.append(result)
            else:
                clusters.append([result])
        
        # Convert clusters to MultiPassResult
        combined: List[MultiPassResult] = []
        
        for cluster in clusters:
            if not cluster:
                continue
            
            # Pick the best text (highest confidence)
            best_result = max(cluster, key=lambda r: r.confidence)
            
            # Calculate combined confidence (boost for multiple detections)
            avg_confidence = sum(r.confidence for r in cluster) / len(cluster)
            vote_bonus = min(0.2, 0.05 * len(cluster))  # Up to +0.2 for 4+ votes
            combined_confidence = min(1.0, avg_confidence + vote_bonus)
            
            # Use average bbox
            avg_x = int(sum(r.bbox[0] for r in cluster) / len(cluster))
            avg_y = int(sum(r.bbox[1] for r in cluster) / len(cluster))
            avg_w = int(sum(r.bbox[2] for r in cluster) / len(cluster))
            avg_h = int(sum(r.bbox[3] for r in cluster) / len(cluster))
            
            combined.append(MultiPassResult(
                text=best_result.text,
                confidence=combined_confidence,
                bbox=(avg_x, avg_y, avg_w, avg_h),
                vote_count=len(cluster),
                sources=[r.preprocessing for r in cluster]
            ))
        
        # Sort by x position (left to right)
        combined.sort(key=lambda r: r.bbox[0])
        
        return combined
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity ratio."""
        if not text1 or not text2:
            return 0.0
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def preprocess_for_ocr(self, image: np.ndarray, 
                           upscale: float = 2.0,
                           enhance_contrast: bool = True) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy (legacy method).
        
        For better results, use read_text_multipass() instead.
        """
        result = image.copy()
        
        if upscale > 1.0:
            h, w = result.shape[:2]
            new_w, new_h = int(w * upscale), int(h * upscale)
            result = cv2.resize(result, (new_w, new_h), 
                               interpolation=cv2.INTER_CUBIC)
        
        if enhance_contrast:
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return result


# Global singleton for reuse
_ocr_engine: Optional[OCREngine] = None


def get_ocr_engine(use_gpu: bool = True) -> OCREngine:
    """Get the global OCR engine instance."""
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = OCREngine(use_gpu=use_gpu)
    return _ocr_engine
