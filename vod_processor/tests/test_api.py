"""
Tests for VOD Processor
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data


def test_upload_invalid_file_type():
    """Test that invalid file types are rejected."""
    # Create a fake text file
    files = {"file": ("test.txt", b"not a video", "text/plain")}
    response = client.post("/api/v1/vod/upload", files=files)
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


def test_get_nonexistent_job():
    """Test getting a job that doesn't exist."""
    response = client.get("/api/v1/vod/nonexistent-job-id/status")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_timeline_incomplete_job():
    """Test that timeline request fails for incomplete jobs."""
    # First need to create a job somehow - skip for now
    pass


class TestKillfeedDetector:
    """Tests for killfeed detection."""
    
    def test_row_segmentation(self):
        """Test that rows are properly segmented."""
        # Would need sample images
        pass
    
    def test_team_color_detection(self):
        """Test team color detection."""
        # Would need sample images
        pass
    
    def test_deduplication(self):
        """Test kill event deduplication."""
        from app.services.vod_processor import KillfeedDetector
        
        detector = KillfeedDetector("killfeed", 8.0)
        
        # Simulate a kill
        detector.recent_signatures.append(
            (1000.0, "teal", "orange", "Chronicle", "Ethan")
        )
        
        # Same kill should be duplicate
        sig = (1500.0, "teal", "orange", "Chronicle", "Ethan")
        assert detector._is_duplicate(1500.0, sig) == True
        
        # Different kill should not be duplicate
        sig2 = (1500.0, "orange", "teal", "s0m", "Boaster")
        assert detector._is_duplicate(1500.0, sig2) == False


class TestROIConfig:
    """Tests for ROI configuration."""
    
    def test_roi_bounds(self):
        """Test that all ROIs are within valid bounds."""
        from config import ROI_CONFIG
        
        for name, (x, y, w, h) in ROI_CONFIG.items():
            assert 0 <= x <= 1, f"ROI {name} x out of bounds"
            assert 0 <= y <= 1, f"ROI {name} y out of bounds"
            assert 0 < w <= 1, f"ROI {name} width out of bounds"
            assert 0 < h <= 1, f"ROI {name} height out of bounds"
            assert x + w <= 1, f"ROI {name} extends past right edge"
            assert y + h <= 1, f"ROI {name} extends past bottom edge"
    
    def test_roi_px_conversion(self):
        """Test ROI coordinate conversion."""
        from app.services.vod_processor import roi_to_px
        
        # Test with 1920x1080 resolution
        px = roi_to_px(1920, 1080, (0.5, 0.5, 0.25, 0.25))
        assert px == (960, 540, 480, 270)
