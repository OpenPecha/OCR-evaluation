"""
Tests for CER_evaluation.py — edit distance, CER computation, normalisation,
loading helpers, and end-to-end evaluation logic.
"""

import logging
from pathlib import Path

import pandas as pd
import pytest

from CER_evaluation import (
    _edit_distance,
    _normalise,
    compute_cer,
    evaluate_model,
    load_benchmark,
    load_model_csv,
    save_evaluation,
    save_summary,
)

logger = logging.getLogger(__name__)


# ── _edit_distance ───────────────────────────────────────────────────────────

class TestEditDistance:
    """Unit tests for the Levenshtein edit distance function."""

    def test_identical_strings(self):
        logger.info("Testing edit distance with identical strings")
        assert _edit_distance("abc", "abc") == 0

    def test_empty_strings(self):
        logger.info("Testing edit distance with empty strings")
        assert _edit_distance("", "") == 0

    def test_one_empty(self):
        logger.info("Testing edit distance with one empty string")
        assert _edit_distance("abc", "") == 3
        assert _edit_distance("", "xyz") == 3

    def test_single_insertion(self):
        assert _edit_distance("abc", "abcd") == 1

    def test_single_deletion(self):
        assert _edit_distance("abcd", "abc") == 1

    def test_single_substitution(self):
        assert _edit_distance("abc", "axc") == 1

    def test_known_distance(self):
        # "kitten" → "sitting" = 3
        logger.info("Testing known edit distance: kitten → sitting")
        assert _edit_distance("kitten", "sitting") == 3

    def test_symmetric(self):
        """Edit distance is symmetric."""
        assert _edit_distance("abc", "xyz") == _edit_distance("xyz", "abc")

    def test_tibetan_characters(self):
        """Verify edit distance works with Tibetan Unicode characters."""
        logger.info("Testing edit distance with Tibetan characters")
        s1 = "བཀྲ་ཤིས"
        s2 = "བཀྲ་ཤིས"
        assert _edit_distance(s1, s2) == 0

        s3 = "བཀྲ་ཤིས"
        s4 = "བཀྲ་ཤི"
        assert _edit_distance(s3, s4) == 1


# ── compute_cer ──────────────────────────────────────────────────────────────

class TestComputeCER:
    """Unit tests for the CER computation."""

    def test_perfect_match(self):
        logger.info("Testing CER: perfect match → 0.0")
        assert compute_cer("hello", "hello") == 0.0

    def test_both_empty(self):
        assert compute_cer("", "") == 0.0

    def test_empty_reference_nonempty_inference(self):
        """By convention CER = 1.0 when reference is empty but inference is not."""
        assert compute_cer("some text", "") == 1.0

    def test_empty_inference_nonempty_reference(self):
        cer = compute_cer("", "hello")
        assert cer == 1.0  # 5 deletions / 5 chars = 1.0

    def test_known_cer(self):
        # "abc" vs "axc" → distance 1, len(ref) 3 → CER ≈ 0.333
        cer = compute_cer("axc", "abc")
        assert abs(cer - 1 / 3) < 1e-6

    def test_cer_bounded(self):
        """CER should be non-negative."""
        cer = compute_cer("completely wrong", "reference text")
        assert cer >= 0.0


# ── _normalise ───────────────────────────────────────────────────────────────

class TestNormalise:
    """Tests for the configurable _normalise() function."""

    # ── Default behaviour (no normalisation) ─────────────────────────────
    def test_default_no_change(self):
        """With no flags, text is returned unchanged."""
        assert _normalise("  hello world  ") == "  hello world  "

    def test_default_preserves_newlines(self):
        assert _normalise("line1\n  line2\n") == "line1\n  line2\n"

    def test_default_empty_string(self):
        assert _normalise("") == ""

    # ── Whitespace normalisation ─────────────────────────────────────────
    def test_whitespace_strips(self):
        assert _normalise("  hello world  ", whitespace=True) == "hello world"

    def test_whitespace_collapses(self):
        assert _normalise("hello   \t  world", whitespace=True) == "hello world"

    def test_whitespace_newlines(self):
        logger.info("Testing whitespace normalisation of newlines")
        assert _normalise("line1\n  line2\n", whitespace=True) == "line1 line2"

    def test_whitespace_already_clean(self):
        assert _normalise("hello world", whitespace=True) == "hello world"

    # ── Unicode normalisation ────────────────────────────────────────────
    def test_unicode_nfc(self):
        import unicodedata
        # é as NFD (e + combining acute) → NFC should compose it
        nfd = unicodedata.normalize("NFD", "é")
        assert _normalise(nfd, unicode_form="NFC") == "é"

    def test_unicode_nfkc(self):
        # ﬁ (LATIN SMALL LIGATURE FI) → fi under NFKC
        assert _normalise("\ufb01", unicode_form="NFKC") == "fi"

    def test_unicode_none_leaves_unchanged(self):
        import unicodedata
        nfd = unicodedata.normalize("NFD", "é")
        assert _normalise(nfd, unicode_form=None) == nfd

    # ── Both flags together ──────────────────────────────────────────────
    def test_combined(self):
        import unicodedata
        nfd = unicodedata.normalize("NFD", "  hé  llo  ")
        result = _normalise(nfd, whitespace=True, unicode_form="NFC")
        assert result == "hé llo"


# ── load_benchmark ───────────────────────────────────────────────────────────

class TestLoadBenchmark:
    def test_loads_correct_entries(self, tmp_benchmark):
        logger.info("Testing benchmark CSV loading")
        benchmark = load_benchmark(tmp_benchmark / "benchmark.csv")
        assert len(benchmark) == 3
        assert ("img_001.png", "batch-1") in benchmark
        assert ("img_003.png", "batch-2") in benchmark
        assert benchmark[("img_001.png", "batch-1")] == "བཀྲ་ཤིས་བདེ་ལེགས"

    def test_handles_whitespace_in_csv(self, tmp_path):
        """Verify that stray whitespace in column names / values is stripped."""
        csv_path = tmp_path / "ws.csv"
        csv_path.write_text(
            " image_name , batch_id , transcript \n"
            " foo.png , b1 , hello \n"
        )
        benchmark = load_benchmark(csv_path)
        assert ("foo.png", "b1") in benchmark


# ── load_model_csv ───────────────────────────────────────────────────────────

class TestLoadModelCSV:
    def test_basic_load(self, tmp_path):
        logger.info("Testing model CSV loading")
        csv_path = tmp_path / "model.csv"
        df = pd.DataFrame([
            {"image_name": "a.png", "batch_id": "b1", "inference": "text1"},
            {"image_name": "b.png", "batch_id": "b1", "inference": "text2"},
        ])
        df.to_csv(csv_path, index=False)

        rows = load_model_csv(csv_path)
        assert len(rows) == 2
        assert rows[0] == ("a.png", "b1", "text1")


# ── evaluate_model ───────────────────────────────────────────────────────────

class TestEvaluateModel:
    def test_perfect_model(self, tmp_path, tmp_benchmark):
        """Model that produces perfect transcriptions → CER 0.0."""
        logger.info("Testing evaluation with perfect model output")
        benchmark = load_benchmark(tmp_benchmark / "benchmark.csv")

        # Write a model CSV with perfect transcriptions
        csv_path = tmp_path / "perfect.csv"
        df = pd.DataFrame([
            {"image_name": "img_001.png", "batch_id": "batch-1",
             "inference": "བཀྲ་ཤིས་བདེ་ལེགས"},
            {"image_name": "img_002.png", "batch_id": "batch-1",
             "inference": "ཨོཾ་མ་ཎི་པདྨེ་ཧཱུྃ"},
            {"image_name": "img_003.png", "batch_id": "batch-2",
             "inference": "སྤྱན་རས་གཟིགས"},
        ])
        df.to_csv(csv_path, index=False)

        result = evaluate_model("perfect_model", csv_path, benchmark)

        assert result["model"] == "perfect_model"
        assert result["overall_cer"] == 0.0
        assert len(result["per_image"]) == 3
        assert all(row["cer"] == 0.0 for row in result["per_image"])
        logger.info("Perfect model evaluation passed — CER: %.4f", result["overall_cer"])

    def test_imperfect_model(self, tmp_path, tmp_benchmark):
        """Model with some errors → CER > 0."""
        logger.info("Testing evaluation with imperfect model output")
        benchmark = load_benchmark(tmp_benchmark / "benchmark.csv")

        csv_path = tmp_path / "imperfect.csv"
        df = pd.DataFrame([
            {"image_name": "img_001.png", "batch_id": "batch-1",
             "inference": "wrong text"},
            {"image_name": "img_002.png", "batch_id": "batch-1",
             "inference": "ཨོཾ་མ་ཎི་པདྨེ་ཧཱུྃ"},
            {"image_name": "img_003.png", "batch_id": "batch-2",
             "inference": "སྤྱན་རས་གཟིགས"},
        ])
        df.to_csv(csv_path, index=False)

        result = evaluate_model("imperfect_model", csv_path, benchmark)

        assert result["overall_cer"] > 0.0
        assert len(result["per_batch"]) == 2
        logger.info("Imperfect model CER: %.4f", result["overall_cer"])

    def test_empty_model_output(self, tmp_path, tmp_benchmark):
        """Model producing empty strings → high CER."""
        benchmark = load_benchmark(tmp_benchmark / "benchmark.csv")

        csv_path = tmp_path / "empty.csv"
        df = pd.DataFrame([
            {"image_name": "img_001.png", "batch_id": "batch-1", "inference": ""},
            {"image_name": "img_002.png", "batch_id": "batch-1", "inference": ""},
            {"image_name": "img_003.png", "batch_id": "batch-2", "inference": ""},
        ])
        df.to_csv(csv_path, index=False)

        result = evaluate_model("empty_model", csv_path, benchmark)
        assert result["overall_cer"] == 1.0
        logger.info("Empty model CER: %.4f (expected 1.0)", result["overall_cer"])


# ── save_evaluation / save_summary ───────────────────────────────────────────

class TestSaveResults:
    def test_save_evaluation_creates_csv(self, tmp_output):
        logger.info("Testing save_evaluation output file")
        result = {
            "model": "test_model",
            "overall_cer": 0.123,
            "per_batch": {"b1": {"cer": 0.1, "count": 2}},
            "per_image": [
                {"image_name": "a.png", "batch_id": "b1", "cer": 0.1},
                {"image_name": "b.png", "batch_id": "b1", "cer": 0.15},
            ],
        }
        csv_path = save_evaluation(result, tmp_output)

        assert csv_path.exists()
        assert csv_path.suffix == ".csv"

        # Verify CSV content
        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert list(df.columns) == ["image_name", "batch_id", "cer"]
        assert df["cer"].iloc[0] == 0.1
        logger.info("Saved evaluation CSV verified at %s", csv_path)

    def test_save_summary_creates_csv(self, tmp_output):
        logger.info("Testing save_summary")
        results = [
            {
                "model": "m1",
                "overall_cer": 0.1,
                "per_batch": {"b1": {"cer": 0.1, "count": 5}},
            },
            {
                "model": "m2",
                "overall_cer": 0.2,
                "per_batch": {"b1": {"cer": 0.2, "count": 5}},
            },
        ]
        summary_path = save_summary(results, tmp_output)
        assert summary_path.exists()

        df = pd.read_csv(summary_path)
        assert len(df) == 2
        assert list(df["model"]) == ["m1", "m2"]
        logger.info("Summary CSV verified at %s", summary_path)
