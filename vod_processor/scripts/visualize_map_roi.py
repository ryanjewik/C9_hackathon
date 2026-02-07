"""
Visualize the Map Detection ROI

Creates a PNG image showing the ROI overlay on a sample frame.
Usage: python visualize_map_roi.py <video_path> [output_path]
"""

import sys
import cv2
import numpy as np
import os

# Map indicator ROI: (x, y, width, height) normalized coordinates
# Series scoreboard in top-left showing "CURRENT: <MAP>"
# Shows: "LOTUS 13-6 | CURRENT: ABYSS | NEXT: ASCENT"
MAP_INDICATOR_ROI = (0.0, 0.0, 0.32, 0.025)


def visualize_roi(video_path: str, output_path: str = None, frame_number: int = 300):
    """
    Extract a frame and draw the map detection ROI on it.
    
    Args:
        video_path: Path to video file
        output_path: Output PNG path (default: map_roi_visualization.png)
        frame_number: Frame to extract (default: 300 = ~10 seconds in)
    """
    if output_path is None:
        output_path = "map_roi_visualization.png"
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return
    
    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Cannot read frame {frame_number}")
        return
    
    h, w = frame.shape[:2]
    
    # Calculate ROI coordinates
    x1 = int(MAP_INDICATOR_ROI[0] * w)
    y1 = int(MAP_INDICATOR_ROI[1] * h)
    x2 = int((MAP_INDICATOR_ROI[0] + MAP_INDICATOR_ROI[2]) * w)
    y2 = int((MAP_INDICATOR_ROI[1] + MAP_INDICATOR_ROI[3]) * h)
    
    # Draw ROI rectangle on frame
    overlay = frame.copy()
    
    # Semi-transparent fill
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Solid border
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    # Add label
    label = f"MAP ROI: ({x1},{y1}) to ({x2},{y2})"
    cv2.putText(frame, label, (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Also extract and save just the ROI (from original frame, without overlay)
    cap2 = cv2.VideoCapture(video_path)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret2, original_frame = cap2.read()
    cap2.release()
    
    roi_crop = original_frame[y1:y2, x1:x2] if ret2 else frame[y1:y2, x1:x2]
    roi_output = output_path.replace(".png", "_crop.png")
    
    # Save full frame with overlay
    cv2.imwrite(output_path, frame)
    print(f"Saved visualization: {output_path}")
    print(f"  Frame size: {w}x{h}")
    print(f"  ROI: x={x1}-{x2}, y={y1}-{y2} ({x2-x1}x{y2-y1} pixels)")
    print(f"  ROI normalized: {MAP_INDICATOR_ROI}")
    
    # Save ROI crop
    if roi_crop.size > 0:
        cv2.imwrite(roi_output, roi_crop)
        print(f"Saved ROI crop: {roi_output}")
        
        # Also save preprocessed version (scaled 3x with contrast enhancement)
        preprocessed = preprocess_roi(roi_crop, scale_factor=3.0)
        preprocessed_output = output_path.replace(".png", "_preprocessed.png")
        cv2.imwrite(preprocessed_output, preprocessed)
        print(f"Saved preprocessed ROI: {preprocessed_output}")


def preprocess_roi(roi: np.ndarray, scale_factor: float = 3.0) -> np.ndarray:
    """
    Preprocess the ROI for better OCR readability.
    """
    # Upscale for better text recognition
    h, w = roi.shape[:2]
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Slight sharpening
    kernel = np.array([[-1, -1, -1],
                      [-1,  9, -1],
                      [-1, -1, -1]])
    gray = cv2.filter2D(gray, -1, kernel)
    
    return gray


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_map_roi.py <video_path> [output_path] [frame_number]")
        print("")
        print("Example:")
        print("  python visualize_map_roi.py match.mp4")
        print("  python visualize_map_roi.py match.mp4 roi_vis.png 500")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "map_roi_visualization.png"
    frame_number = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    
    visualize_roi(video_path, output_path, frame_number)


if __name__ == "__main__":
    main()
