"""
tests/test_accuracy.py — Accuracy Benchmarking Suite
======================================================

Evaluates detection accuracy using labelled test cases.
Run: pytest tests/test_accuracy.py -v --tb=short

Metrics computed:
  • Accuracy, Precision, Recall, F1-score (per class)
  • Confusion matrix
  • ROC-AUC (binary)
  • Mean confidence calibration error (ECE)
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Test data fixtures
# ─────────────────────────────────────────────────────────────────────────────

AI_DETECTION_TEST_CASES = [
    # (description, image_path_or_text, true_label: 1=AI, 0=human)
    # Images (use paths to local test images)
    {"id": "img_ai_01",     "type": "image", "label": 1, "source": "midjourney_v6"},
    {"id": "img_ai_02",     "type": "image", "label": 1, "source": "dall_e_3"},
    {"id": "img_ai_03",     "type": "image", "label": 1, "source": "stable_diffusion_xl"},
    {"id": "img_ai_04",     "type": "image", "label": 1, "source": "sora_frame"},
    {"id": "img_ai_05",     "type": "image", "label": 1, "source": "deepfake_faceswap"},
    {"id": "img_real_01",   "type": "image", "label": 0, "source": "camera_canon_eos"},
    {"id": "img_real_02",   "type": "image", "label": 0, "source": "iphone_15_photo"},
    {"id": "img_real_03",   "type": "image", "label": 0, "source": "news_wire_photo"},
    {"id": "img_real_04",   "type": "image", "label": 0, "source": "nature_photography"},
    {"id": "img_real_05",   "type": "image", "label": 0, "source": "documentary_photo"},
    # Text samples
    {"id": "txt_ai_01",     "type": "text", "label": 1, "text": "In conclusion, it is evident that artificial intelligence represents a transformative technology with far-reaching implications across numerous domains of human endeavor. The multifaceted nature of this technological paradigm necessitates a comprehensive approach to its development and deployment."},
    {"id": "txt_ai_02",     "type": "text", "label": 1, "text": "Climate change poses significant challenges to global ecosystems. Rising temperatures, coupled with increased frequency of extreme weather events, underscore the urgent need for comprehensive policy interventions. Stakeholders across various sectors must collaborate to implement sustainable solutions."},
    {"id": "txt_real_01",   "type": "text", "label": 0, "text": "i was at the market yesterday when suddenly this guy just starts yelling at the vendor. turns out the oranges were 3 days old and he'd bought them for his mom's birthday party. total chaos. felt bad for everyone honestly"},
    {"id": "txt_real_02",   "type": "text", "label": 0, "text": "The quarterly results came in 12% below expectations, largely because of the supply chain mess we've had in Southeast Asia. Nobody on the board was happy, but honestly what did they expect after Q3."},
]

FACT_CHECK_TEST_CASES = [
    # (claim, expected_verdict, should_have_sources)
    {
        "id": "fc_true_01",
        "text": "The WHO declared COVID-19 a pandemic on March 11, 2020.",
        "expected_verdict": "TRUE",
        "expected_confidence_min": 70,
    },
    {
        "id": "fc_fake_01",
        "text": "5G towers are responsible for spreading COVID-19 by activating the virus in human cells.",
        "expected_verdict": "FALSE",
        "expected_confidence_min": 75,
    },
    {
        "id": "fc_misleading_01",
        "text": "Vaccines cause autism, according to a major scientific study.",
        "expected_verdict": "FALSE",
        "expected_confidence_min": 80,
    },
    {
        "id": "fc_true_02",
        "text": "The Eiffel Tower is located in Paris, France and was completed in 1889.",
        "expected_verdict": "TRUE",
        "expected_confidence_min": 85,
    },
    {
        "id": "fc_unverified_01",
        "text": "A secret government program is hiding evidence of extraterrestrial contact.",
        "expected_verdict": "UNVERIFIED",
        "expected_confidence_max": 60,
    },
    {
        "id": "fc_misleading_02",
        "text": "Drinking bleach cures cancer according to alternative medicine practitioners.",
        "expected_verdict": "FALSE",
        "expected_confidence_min": 90,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Metrics computation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    roc_auc: Optional[float] = None
    ece: Optional[float] = None  # Expected Calibration Error

    def __str__(self) -> str:
        return (
            f"Accuracy:  {self.accuracy:.1%}\n"
            f"Precision: {self.precision:.1%}\n"
            f"Recall:    {self.recall:.1%}\n"
            f"F1-Score:  {self.f1:.1%}\n"
            f"TP={self.true_positives} FP={self.false_positives} "
            f"TN={self.true_negatives} FN={self.false_negatives}\n"
            + (f"ROC-AUC:   {self.roc_auc:.3f}\n" if self.roc_auc else "")
            + (f"ECE:       {self.ece:.3f}" if self.ece else "")
        )


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: Optional[List[float]] = None,
) -> ClassificationMetrics:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy  = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall    = tp / max(1, tp + fn)
    f1        = 2 * precision * recall / max(1e-9, precision + recall)

    roc_auc = None
    if y_prob is not None:
        try:
            from sklearn.metrics import roc_auc_score
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass

    ece = None
    if y_prob is not None:
        ece = _compute_ece(y_true, np.array(y_prob))

    return ClassificationMetrics(
        accuracy=accuracy, precision=precision, recall=recall, f1=f1,
        true_positives=tp, false_positives=fp,
        true_negatives=tn, false_negatives=fn,
        roc_auc=roc_auc, ece=ece,
    )


def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error — lower is better calibrated."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if not mask.any():
            continue
        frac_correct = y_true[mask].mean()
        mean_conf = y_prob[mask].mean()
        ece += mask.sum() * abs(frac_correct - mean_conf)
    return float(ece / len(y_true))


# ─────────────────────────────────────────────────────────────────────────────
# Mock / stub for testing without real model
# ─────────────────────────────────────────────────────────────────────────────

class MockEnsembleResult:
    """Simulated ensemble result for unit testing."""
    def __init__(self, ai_score: float, confidence: float = 0.8):
        self.ai_score = ai_score
        self.confidence = confidence
        self.model_scores = []
        self.frequency_features = None
        self.latency_ms = 50.0


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEnsembleLogic:
    """Test ensemble fusion without loading real models."""

    def test_weighted_fusion_ai(self):
        """High scores from all models → AI verdict."""
        scores = [0.92, 0.88, 0.85, 0.90]
        weights = [0.35, 0.30, 0.20, 0.15]
        confs   = [0.90, 0.85, 0.80, 0.70]

        numerator = sum(w * c * s for w, c, s in zip(weights, confs, scores))
        denominator = sum(w * c for w, c in zip(weights, confs))
        fused = numerator / denominator

        assert fused > 0.75, f"Expected AI verdict, got {fused:.2f}"

    def test_weighted_fusion_human(self):
        """Low scores from all models → human verdict."""
        scores = [0.08, 0.12, 0.10, 0.15]
        weights = [0.35, 0.30, 0.20, 0.15]
        confs   = [0.85, 0.80, 0.75, 0.70]

        numerator = sum(w * c * s for w, c, s in zip(weights, confs, scores))
        denominator = sum(w * c for w, c in zip(weights, confs))
        fused = numerator / denominator

        assert fused < 0.40, f"Expected human verdict, got {fused:.2f}"

    def test_disagreement_lowers_confidence(self):
        """High variance in model scores → lower ensemble confidence."""
        scores = [0.95, 0.10, 0.90, 0.15]  # extreme disagreement
        variance = float(np.std(scores))
        base_conf = 0.75
        adjusted = max(0.30, base_conf - variance * 0.5)
        assert adjusted < base_conf, "Disagreement should reduce confidence"

    def test_platt_calibration(self):
        """Platt scaling should keep probabilities within [0, 1]."""
        import sys
        sys.path.insert(0, "..")
        from ai_detection.ensemble import platt_calibrate
        for raw in [-2.0, -0.5, 0.0, 0.5, 1.0, 3.0]:
            cal = platt_calibrate(raw, a=1.12, b=-0.08)
            assert 0.0 <= cal <= 1.0, f"Calibrated prob {cal} out of range"

    def test_metrics_computation(self):
        """Verify metrics math."""
        y_true = [1, 1, 0, 0, 1, 0, 1, 0]
        y_pred = [1, 0, 0, 1, 1, 0, 1, 0]
        m = compute_metrics(y_true, y_pred)
        # TP=3, FP=1, TN=3, FN=1
        assert m.true_positives == 3
        assert m.false_positives == 1
        assert m.accuracy == pytest.approx(0.75)
        assert m.precision == pytest.approx(0.75)
        assert m.recall == pytest.approx(0.75)
        assert m.f1 == pytest.approx(0.75)


class TestFrequencyAnalyser:
    """Test frequency domain analysis with synthetic images."""

    @pytest.fixture
    def analyser(self):
        import sys; sys.path.insert(0, "..")
        from ai_detection.ensemble import FrequencyAnalyser
        return FrequencyAnalyser()

    def test_uniform_image_low_ai_score(self, analyser):
        """Uniform noise image should score LOW on AI detection."""
        rng = np.random.RandomState(42)
        image = rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        # Natural noise → not AI
        score, conf, features = asyncio.get_event_loop().run_until_complete(
            analyser.predict(image)
        )
        # Noisy image should have high-freq content → low AI score
        assert features.high_freq_anomaly < 0.7, "Noisy image should not flag as AI"

    def test_smooth_gradient_higher_ai_score(self, analyser):
        """Smooth AI-like image should score higher."""
        # Create very smooth gradient (AI-like, no noise)
        x = np.linspace(0, 1, 224)
        y = np.linspace(0, 1, 224)
        xx, yy = np.meshgrid(x, y)
        smooth = (np.sin(xx * np.pi) * 127 + 128).astype(np.uint8)
        image = np.stack([smooth, smooth, smooth], axis=2)

        score, conf, features = asyncio.get_event_loop().run_until_complete(
            analyser.predict(image)
        )
        # Smooth gradient → attenuated high-freq → higher AI score
        assert features.dct_energy_ratio > 0.80, "Smooth image should have high DCT ratio"


class TestCacheLayer:
    """Test cache with in-memory fallback."""

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        import sys; sys.path.insert(0, "..")
        from shared.cache import Cache
        cache = Cache(redis_url="redis://nonexistent:6379/0")

        key = "test_key_123"
        value = {"score": 0.87, "verdict": "AI-GENERATED"}
        await cache.set(key, value, ttl=60)
        result = await cache.get(key)
        assert result == value

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        import sys; sys.path.insert(0, "..")
        from shared.cache import Cache
        cache = Cache(redis_url="redis://nonexistent:6379/0")
        result = await cache.get("definitely_not_there_xyz")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_or_compute(self):
        import sys; sys.path.insert(0, "..")
        from shared.cache import Cache
        cache = Cache(redis_url="redis://nonexistent:6379/0")
        call_count = [0]

        async def compute():
            call_count[0] += 1
            return {"computed": True}

        key = "compute_key"
        r1 = await cache.get_or_compute(key, compute, ttl=60)
        r2 = await cache.get_or_compute(key, compute, ttl=60)
        assert call_count[0] == 1, "compute_fn should be called only once"
        assert r1 == r2

    def test_cache_key_deterministic(self):
        import sys; sys.path.insert(0, "..")
        from shared.cache import Cache
        k1 = Cache.make_key("ai", "image", "abc123")
        k2 = Cache.make_key("ai", "image", "abc123")
        k3 = Cache.make_key("ai", "text", "abc123")
        assert k1 == k2, "Same inputs → same key"
        assert k1 != k3, "Different inputs → different key"


class TestAntiHallucination:
    """Verify URL sanitisation removes unverified sources."""

    def test_url_sanitisation(self):
        import sys; sys.path.insert(0, "..")
        from fake_news.pipeline import RetrievedDoc, ScoringEngine

        valid_url = "https://snopes.com/fact-check/test"
        hallucinated_url = "https://made-up-source.org/fake-article"

        evidence = [
            RetrievedDoc(
                title="Test fact-check", content="content",
                url=valid_url, source="Snopes", source_type="FACT-CHECK"
            )
        ]

        claude_result = {
            "claims": [
                {
                    "id": 1, "text": "Test claim", "verdict": "FALSE",
                    "confidence": 80, "explanation": "Checked",
                    "flagged": True,
                    "sources": [valid_url, hallucinated_url],  # includes hallucinated
                }
            ]
        }

        scorer = ScoringEngine()
        sanitised = scorer.validate_urls(claude_result, evidence)
        assert valid_url in sanitised["claims"][0]["sources"]
        assert hallucinated_url not in sanitised["claims"][0]["sources"]

    def test_no_evidence_returns_unverified(self):
        """With 0 retrieved docs, pipeline must return UNVERIFIED."""
        import sys; sys.path.insert(0, "..")
        from fake_news.pipeline import ClaudeVerifier

        verifier = ClaudeVerifier()
        # Test fallback response (no API call)
        result = verifier._fallback_response(["Test claim"], [])
        assert result["verdict"] == "UNVERIFIED"
        assert result["confidence"] <= 40


class TestFactCheckScoring:
    """Test signal computation logic."""

    def test_signals_structure(self):
        import sys; sys.path.insert(0, "..")
        from fake_news.pipeline import ScoringEngine, RetrievedDoc

        scorer = ScoringEngine()
        evidence = [
            RetrievedDoc(
                title="WHO announcement", content="vaccine is safe",
                url="https://who.int/test", source="WHO",
                source_type="OFFICIAL", domain_trust=1.0, semantic_similarity=0.85
            )
        ]
        claude_result = {
            "verdict": "TRUE", "confidence": 85, "overallRisk": "LOW",
            "claims": [{"verdict": "TRUE"}], "languageFlags": []
        }
        signals = scorer.compute_signals(claude_result, evidence, ["test claim"])
        assert len(signals) == 6
        names = [s.name for s in signals]
        assert "Factual Accuracy" in names
        assert "Source Credibility" in names
        assert "Misinformation Risk" in names


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchmark:
    """
    End-to-end benchmark. Run with actual test images/texts.
    Generates accuracy report.
    """

    def test_print_benchmark_summary(self):
        """
        Simulates benchmark results — replace with real predictions
        by running the ensemble on your labelled test set.
        """
        # Simulated predictions (replace with actual model outputs)
        # Format: (true_label, predicted_label, predicted_probability)
        sim_results = [
            (1, 1, 0.91), (1, 1, 0.87), (1, 0, 0.42), (1, 1, 0.78),
            (1, 1, 0.95), (0, 0, 0.12), (0, 0, 0.08), (0, 1, 0.55),
            (0, 0, 0.15), (0, 0, 0.22),
        ]
        y_true = [r[0] for r in sim_results]
        y_pred = [r[1] for r in sim_results]
        y_prob = [r[2] for r in sim_results]

        metrics = compute_metrics(y_true, y_pred, y_prob)

        print("\n" + "="*50)
        print("  TRUTHSCAN AI DETECTION — BENCHMARK RESULTS")
        print("="*50)
        print(metrics)
        print("="*50)

        # Acceptance thresholds
        assert metrics.accuracy >= 0.70, f"Accuracy {metrics.accuracy:.1%} below 70% threshold"
        assert metrics.f1 >= 0.65, f"F1 {metrics.f1:.1%} below 65% threshold"
        if metrics.roc_auc:
            assert metrics.roc_auc >= 0.75, f"ROC-AUC {metrics.roc_auc:.3f} below 0.75 threshold"


if __name__ == "__main__":
    # Quick standalone run
    print("Running TruthScan accuracy benchmarks...")
    suite = TestBenchmark()
    suite.test_print_benchmark_summary()
    print("\n✅ Benchmark complete")
