"""
Property-based and unit tests for crimai/face_engine.py helpers.

This file holds tasks 4.5, 4.6, 4.7, and 4.8 tests.

insightface and onnxruntime are heavy ML dependencies not available in CI.
We stub them in sys.modules before importing face_engine so that the module
loads without errors; match_embedding and _to_relative do not call into
those libraries at runtime.
"""

from __future__ import annotations

import sys
import types

# ---------------------------------------------------------------------------
# Stub heavy ML dependencies before importing face_engine
# ---------------------------------------------------------------------------

def _make_stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


if "insightface" not in sys.modules:
    insightface_mod = _make_stub_module("insightface")
    insightface_app = _make_stub_module("insightface.app")

    class _FaceAnalysisStub:
        pass

    insightface_app.FaceAnalysis = _FaceAnalysisStub
    insightface_mod.app = insightface_app

if "onnxruntime" not in sys.modules:
    ort_mod = _make_stub_module("onnxruntime")
    ort_mod.get_available_providers = lambda: []

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from crimai.config import SIMILARITY_THRESH
from crimai.face_engine import _to_relative, match_embedding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 512
_BYTES = _DIM * 4  # 4 bytes per float32


def _normalise(arr: np.ndarray) -> np.ndarray:
    """L2-normalise a 1-D float32 array; return a unit vector or zero vector."""
    norm = np.linalg.norm(arr)
    if norm == 0.0:
        return arr
    return arr / norm


def _bytes_to_embedding(raw: bytes) -> np.ndarray:
    """Convert raw bytes to a L2-normalised 512-dim float32 embedding.

    NaN/Inf values are replaced with 0.0 before normalisation so that the
    resulting vector is always finite and suitable for cosine similarity.
    """
    arr = np.frombuffer(raw, dtype=np.float32).copy()
    # Replace non-finite values so normalisation is well-defined
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return _normalise(arr)


# Strategy: a single L2-normalised 512-dim float32 embedding generated from
# raw bytes (much faster than st.lists(st.floats(...), min_size=512)).
_embedding_st = st.binary(min_size=_BYTES, max_size=_BYTES).map(_bytes_to_embedding)

# Strategy: a gallery dict mapping 1-3 label strings to lists of 1-3 embeddings
_gallery_st = st.dictionaries(
    keys=st.text(min_size=1, max_size=20),
    values=st.lists(_embedding_st, min_size=1, max_size=3),
    min_size=1,
    max_size=3,
)


# ---------------------------------------------------------------------------
# Feature: crimai-flask-app, Property 4: Match Threshold Enforcement
# ---------------------------------------------------------------------------

# Validates: Requirements 7.4
@settings(
    max_examples=20,
    suppress_health_check=[HealthCheck.large_base_example, HealthCheck.data_too_large],
)
@given(emb=_embedding_st, gallery=_gallery_st)
def test_match_threshold_enforcement(emb: np.ndarray, gallery: dict) -> None:
    """Property 4: label is non-None iff max cosine similarity >= SIMILARITY_THRESH.

    For any face embedding and any set of suspect galleries, match_embedding()
    SHALL return a non-None label if and only if the maximum cosine similarity
    across all gallery embeddings is >= SIMILARITY_THRESH.
    """
    label, score = match_embedding(emb, gallery)

    # Compute expected max cosine similarity manually
    max_sim = max(
        float(np.dot(emb, g_emb))
        for embeddings in gallery.values()
        for g_emb in embeddings
    )

    # Core property: label is non-None iff max_sim >= SIMILARITY_THRESH
    assert (label is not None) == (max_sim >= SIMILARITY_THRESH), (
        f"label={label!r}, score={score}, max_sim={max_sim}, "
        f"SIMILARITY_THRESH={SIMILARITY_THRESH}"
    )


# ---------------------------------------------------------------------------
# Feature: crimai-flask-app, Property 5: Cosine Similarity Bounds
# ---------------------------------------------------------------------------

# Validates: Requirements 7.4
@settings(
    max_examples=20,
    suppress_health_check=[HealthCheck.large_base_example, HealthCheck.data_too_large],
)
@given(emb=_embedding_st, gallery=_gallery_st)
def test_cosine_similarity_bounds(emb, gallery):
    """Property 5: The score returned by match_embedding() is always in [-1.0, 1.0].

    For any two L2-normalised float32 embeddings, the cosine similarity
    (dot product of unit vectors) must be in [-1.0, 1.0].
    """
    label, score = match_embedding(emb, gallery)
    assert -1.0 <= score <= 1.0, f"score={score} is out of bounds [-1.0, 1.0]"


# ---------------------------------------------------------------------------
# Feature: crimai-flask-app, Property 3: Relative Path Stripping
# ---------------------------------------------------------------------------

# Validates: Requirements 18.5, 18.6
@settings(max_examples=20)
@given(filename=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='._-/')).filter(lambda f: f.strip('/') != ''))
def test_relative_path_stripping(filename):
    """Property 3: _to_relative() strips the static prefix and prepending it reconstructs the original.

    For any path that begins with the static folder prefix:
    1. _to_relative() returns a path that does NOT begin with the static prefix
    2. Prepending STATIC_FOLDER + "/" to the result reconstructs the original path
    """
    import os

    from crimai.config import STATIC_FOLDER

    # Construct a path with the static prefix
    path_with_prefix = STATIC_FOLDER + "/" + filename

    result = _to_relative(path_with_prefix)

    # Property: result does not start with the static prefix
    assert not result.startswith(STATIC_FOLDER + "/"), f"result {result!r} still starts with static prefix"

    # Property: prepending the prefix reconstructs the original (modulo path normalisation)
    reconstructed = STATIC_FOLDER + "/" + result
    # Normalise both for comparison (handle OS path separators)
    assert os.path.normpath(reconstructed).replace(os.sep, "/") == os.path.normpath(path_with_prefix).replace(os.sep, "/")


# ---------------------------------------------------------------------------
# Unit tests for match_embedding() — Task 4.8
# ---------------------------------------------------------------------------

# Validates: Requirements 7.4
class TestMatchEmbedding:
    def test_empty_gallery_returns_none_zero(self):
        """match_embedding() returns (None, 0.0) when gallery is empty."""
        emb = np.ones(512, dtype=np.float32) / np.sqrt(512)
        label, score = match_embedding(emb, {})
        assert label is None
        assert score == 0.0

    def test_single_entry_above_threshold_returns_label(self):
        """match_embedding() returns the correct label when a single gallery entry exceeds the threshold."""
        # Create a unit vector embedding
        emb = np.zeros(512, dtype=np.float32)
        emb[0] = 1.0  # unit vector along first axis

        # Gallery with same vector (cosine similarity = 1.0, which is >= SIMILARITY_THRESH=0.5)
        gallery = {"1:Alice": [emb.copy()]}
        label, score = match_embedding(emb, gallery)
        assert label == "1:Alice"
        assert score >= SIMILARITY_THRESH

    def test_single_entry_below_threshold_returns_none(self):
        """match_embedding() returns None label when similarity is below threshold."""
        emb = np.zeros(512, dtype=np.float32)
        emb[0] = 1.0  # unit vector along first axis

        # Gallery with orthogonal vector (cosine similarity = 0.0, which is < SIMILARITY_THRESH=0.5)
        gallery_emb = np.zeros(512, dtype=np.float32)
        gallery_emb[1] = 1.0  # orthogonal unit vector
        gallery = {"1:Bob": [gallery_emb]}
        label, score = match_embedding(emb, gallery)
        assert label is None
        assert score < SIMILARITY_THRESH


# ---------------------------------------------------------------------------
# Unit tests for _to_relative() — Task 4.8
# ---------------------------------------------------------------------------

# Validates: Requirements 18.5
class TestToRelative:
    def test_strips_static_prefix(self):
        """_to_relative() strips the 'static/' prefix."""
        result = _to_relative("static/uploads/evidence/abc.mp4")
        assert result == "uploads/evidence/abc.mp4"

    def test_already_relative_is_noop(self):
        """_to_relative() is a no-op for paths that don't contain 'static/'."""
        result = _to_relative("uploads/evidence/abc.mp4")
        assert result == "uploads/evidence/abc.mp4"

    def test_nested_path(self):
        """_to_relative() handles nested paths correctly."""
        result = _to_relative("static/uploads/crops/face_001.jpg")
        assert result == "uploads/crops/face_001.jpg"

    def test_absolute_path_with_static_segment(self):
        """_to_relative() strips static/ from an absolute path."""
        import os
        # Simulate an absolute path containing static/
        abs_path = os.path.join(os.getcwd(), "static", "uploads", "output", "video.mp4")
        result = _to_relative(abs_path)
        assert result == "uploads/output/video.mp4"
