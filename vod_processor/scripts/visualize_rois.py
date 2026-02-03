#!/usr/bin/env python3
"""
ROI Visualizer - Draws the ROI regions on a sample frame for debugging.
"""

import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ROI_CONFIG


# Color palette for different ROI regions
COLORS = {
    "minimap": (0, 255, 0),      # Green
    "top_hud": (255, 255, 0),    # Cyan
    "killfeed": (0, 0, 255),     # Red
    "bottom_hud": (255, 0, 255), # Magenta
    "left_player": (255, 165, 0), # Orange
    "right_player": (255, 0, 0),  # Blue
    "default": (200, 200, 200),   # Gray
}


def get_color(roi_name: str) -> tuple:
    """Get color for a ROI based on its name."""
    for key, color in COLORS.items():
        if key in roi_name:
            return color
    return COLORS["default"]


def visualize_rois(video_path: str, output_path: str = None, frame_number: int = 100):
    """
    Draw ROI regions on a sample frame from the video.
    
    Args:
        video_path: Path to the video file
        output_path: Path to save the annotated frame (optional)
        frame_number: Which frame to use
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return
    
    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not read frame {frame_number}")
        return
    
    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}")
    
    # Draw each ROI
    for roi_name, (rx, ry, rw, rh) in ROI_CONFIG.items():
        # Convert normalized to pixel coordinates
        px = int(rx * w)
        py = int(ry * h)
        pw = int(rw * w)
        ph = int(rh * h)
        
        color = get_color(roi_name)
        
        # Draw rectangle
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), color, 2)
        
        # Draw label
        label = roi_name
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Background for text
        cv2.rectangle(frame, (px, py - label_h - 4), (px + label_w + 4, py), color, -1)
        cv2.putText(frame, label, (px + 2, py - 2), font, font_scale, (0, 0, 0), thickness)
    
    # Add legend
    y_offset = 30
    for roi_type, color in COLORS.items():
        if roi_type != "default":
            cv2.rectangle(frame, (10, y_offset - 15), (25, y_offset), color, -1)
            cv2.putText(frame, roi_type, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
    
    # Save or display
    if output_path:
        cv2.imwrite(output_path, frame)
        print(f"Saved annotated frame to: {output_path}")
    else:
        cv2.imshow("ROI Visualization", frame)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_rois.py <video_path> [output_path] [frame_number]")
        print("\nExample:")
        print("  python visualize_rois.py match_vod.mp4 roi_debug.png 500")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    frame_number = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    
    visualize_rois(video_path, output_path, frame_number)


if __name__ == "__main__":
    main()
