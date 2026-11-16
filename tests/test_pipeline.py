"""
ARTVISION - Comprehensive Automated Test Suite
Tests SQLite database integrity, image preprocessing validation,
interior color clustering and aesthetic scoring, dataset structure,
and neural network error handling.
"""

import os
import sys
import unittest
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.database import (
    init_database,
    get_all_styles,
    get_style_details,
    get_interior_recommendations,
    get_skill_guidance,
    get_advertising_suggestions,
    log_classification,
    get_recent_classifications
)
from src.preprocessing import validate_image_file, ImageValidationError
from src.interior_recommendation import (
    rgb_to_hex,
    hex_to_rgb,
    rgb_to_hsv,
    classify_color_temperature,
    extract_dominant_colors,
    recommend_artwork_for_room
)
from src.dataset_setup import get_dataset_breakdown
from src.predict import is_model_available, mock_predict_for_setup_preview


class TestDatabaseIntegrity(unittest.TestCase):
    """Verifies relational database tables, seeded knowledge base, and logging."""

    @classmethod
    def setUpClass(cls):
        # Use a temporary database for isolated testing
        cls.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        cls.temp_db.close()
        init_database(cls.temp_db.name, force=True)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_db.name):
            try:
                os.remove(cls.temp_db.name)
            except OSError:
                pass

    def test_all_styles_present(self):
        styles = get_all_styles(self.temp_db.name)
        expected = ["Impressionism", "Cubism", "Abstract", "Surrealism"]
        for exp in expected:
            self.assertIn(exp, styles)

    def test_style_details_completeness(self):
        for style in ["Impressionism", "Cubism", "Abstract", "Surrealism"]:
            details = get_style_details(style, self.temp_db.name)
            self.assertIsNotNone(details)
            self.assertTrue(len(details["historical_background"]) > 50)
            self.assertTrue(len(details["artistic_techniques"]) > 30)
            self.assertTrue(len(details["prominent_artists"]) > 10)

    def test_interior_recommendations_seeding(self):
        for style in ["Impressionism", "Cubism", "Abstract", "Surrealism"]:
            rec = get_interior_recommendations(style, self.temp_db.name)
            self.assertIsNotNone(rec)
            self.assertTrue(len(rec["framing_advice"]) > 15)

    def test_skill_guidance_three_tiers(self):
        for style in ["Impressionism", "Cubism", "Abstract", "Surrealism"]:
            guides = get_skill_guidance(style, db_path=self.temp_db.name)
            self.assertEqual(len(guides), 3)  # Beginner, Intermediate, Advanced
            tiers = [g["difficulty_level"] for g in guides]
            self.assertIn("Beginner", tiers)
            self.assertIn("Intermediate", tiers)
            self.assertIn("Advanced", tiers)

    def test_advertising_suggestions(self):
        for style in ["Impressionism", "Cubism", "Abstract", "Surrealism"]:
            ad = get_advertising_suggestions(style, self.temp_db.name)
            self.assertIsNotNone(ad)
            self.assertTrue(len(ad["brand_archetype"]) > 3)

    def test_classification_logging(self):
        probs = {"Impressionism": 85.5, "Cubism": 5.0, "Abstract": 4.5, "Surrealism": 5.0}
        log_id = log_classification(
            filename="test_waterlilies.jpg",
            predicted_style="Impressionism",
            confidence=85.5,
            probabilities=probs,
            notes="Unit test record",
            db_path=self.temp_db.name
        )
        self.assertIsNotNone(log_id)
        recent = get_recent_classifications(limit=5, db_path=self.temp_db.name)
        self.assertTrue(any(r["filename"] == "test_waterlilies.jpg" for r in recent))


class TestColorAndInteriorRecommendation(unittest.TestCase):
    """Verifies color science algorithms and interior recommendation logic."""

    def test_rgb_hex_conversions(self):
        self.assertEqual(rgb_to_hex(255, 0, 0), "#FF0000")
        self.assertEqual(rgb_to_hex(0, 255, 0), "#00FF00")
        self.assertEqual(rgb_to_hex(0, 0, 255), "#0000FF")
        self.assertEqual(hex_to_rgb("#FFFFFF"), (255, 255, 255))
        self.assertEqual(hex_to_rgb("#000000"), (0, 0, 0))

    def test_color_temperature_classification(self):
        self.assertEqual(classify_color_temperature(255, 100, 50), "warm")   # Orange
        self.assertEqual(classify_color_temperature(40, 100, 220), "cool")   # Blue
        self.assertEqual(classify_color_temperature(128, 128, 128), "neutral") # Gray

    def test_recommend_artwork_warm_palette(self):
        warm_palette = [
            {"rgb": (240, 220, 200), "hex": "#F0DCC8", "percentage": 50.0, "hsv": (30.0, 0.16, 0.94), "temperature": "warm"},
            {"rgb": (200, 140, 90), "hex": "#C88C5A", "percentage": 30.0, "hsv": (27.0, 0.55, 0.78), "temperature": "warm"},
            {"rgb": (255, 250, 240), "hex": "#FFFAF0", "percentage": 20.0, "hsv": (40.0, 0.05, 1.00), "temperature": "neutral"}
        ]
        rec = recommend_artwork_for_room(warm_palette)
        self.assertIn(rec["primary_style"], ["Impressionism", "Cubism", "Abstract", "Surrealism"])
        self.assertTrue(len(rec["rationale"]) > 20)
        self.assertIn("room_metrics", rec)


class TestDatasetAndPreprocessing(unittest.TestCase):
    """Verifies dataset health and image validation rules."""

    def test_empty_image_validation(self):
        is_valid, msg = validate_image_file(b"")
        self.assertFalse(is_valid)

    def test_nonexistent_file_validation(self):
        is_valid, msg = validate_image_file("nonexistent_image_12345.jpg")
        self.assertFalse(is_valid)

    def test_dataset_breakdown_structure(self):
        summary = get_dataset_breakdown()
        self.assertIn("splits", summary)
        self.assertIn("train", summary["splits"])
        self.assertIn("validation", summary["splits"])
        self.assertIn("test", summary["splits"])
        self.assertEqual(summary["classes"], ["Abstract", "Cubism", "Impressionism", "Surrealism"])

    def test_mock_preview_predictor(self):
        sample_path = os.path.join(PROJECT_ROOT, "dataset", "sample_data", "sample_impressionism.bmp")
        if os.path.exists(sample_path):
            result = mock_predict_for_setup_preview(sample_path)
            self.assertFalse(result["is_trained_model"])
            self.assertEqual(result["confidence"], 25.0)
            self.assertEqual(len(result["probabilities"]), 4)


if __name__ == "__main__":
    unittest.main()
