"""
Integration tests for enroll_single() in crimai/face_engine.py.

insightface and onnxruntime are heavy ML dependencies not available in CI.
We stub them in sys.modules before importing face_engine so that the module
loads without errors.

Validates: Requirements 3.6
"""

from __future__ import annotations

import pickle
import sys
import tempfile
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

import os
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from flask import Flask

from crimai.face_engine import enroll_single
from crimai.models import Suspect, db, login_manager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    """Minimal Flask app with an in-memory SQLite database."""
    _app = Flask(__name__)
    _app.config["TESTING"] = True
    _app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite://"
    _app.config["SECRET_KEY"] = "test"

    db.init_app(_app)
    login_manager.init_app(_app)

    with _app.app_context():
        db.create_all()
        yield _app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def temp_image(tmp_path):
    """Create a valid BGR image file and return its absolute path."""
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    # Draw a simple face-like pattern so the image is non-trivial
    img[20:100, 20:100] = [200, 180, 160]
    img_path = str(tmp_path / "suspect_photo.jpg")
    cv2.imwrite(img_path, img)
    return img_path


@pytest.fixture
def fake_face():
    """Return a mock face object with a 512-dim embedding and a bbox."""
    face = MagicMock()
    face.embedding = np.ones(512, dtype=np.float32)
    face.bbox = [10, 10, 100, 100]
    return face


@pytest.fixture
def mock_face_engine(fake_face):
    """Return a mock FaceAnalysis object that always returns one fake face."""
    fa = MagicMock()
    fa.det_thresh = 0.35
    fa.get.return_value = [fake_face]
    return fa


# ---------------------------------------------------------------------------
# Integration test — Task 6.3
# Validates: Requirements 3.6
# ---------------------------------------------------------------------------

class TestEnrollSingle:
    """Integration tests for enroll_single() with a mocked FaceEngine."""

    def test_enroll_single_sets_status_ready(self, app, temp_image, mock_face_engine):
        """After enroll_single(), suspect.enroll_status must be 'ready'.

        Validates: Requirements 3.6
        """
        with app.app_context():
            # photo_path is stored relative to STATIC_FOLDER ("static").
            # We bypass the path reconstruction by patching cv2.imread to
            # return our test image directly, so photo_path can be any string.
            suspect = Suspect(
                name="Test Suspect",
                region="North",
                photo_path="uploads/suspects/suspect_photo.jpg",
                enroll_status="processing",
            )
            db.session.add(suspect)
            db.session.commit()
            suspect_id = suspect.id

        # Load the real image so we can return it from the cv2.imread mock
        real_img = cv2.imread(temp_image)

        with patch("crimai.face_engine.get_app", return_value=mock_face_engine), \
             patch("crimai.face_engine.cv2.imread", return_value=real_img):
            enroll_single(app, suspect_id)

        with app.app_context():
            updated = db.session.get(Suspect, suspect_id)
            assert updated.enroll_status == "ready", (
                f"Expected enroll_status='ready', got {updated.enroll_status!r}. "
                f"error_msg={updated.error_msg!r}"
            )

    def test_enroll_single_gallery_is_not_none(self, app, temp_image, mock_face_engine):
        """After enroll_single(), suspect.gallery must be non-None.

        Validates: Requirements 3.6
        """
        with app.app_context():
            suspect = Suspect(
                name="Gallery Suspect",
                region="South",
                photo_path="uploads/suspects/suspect_photo.jpg",
                enroll_status="processing",
            )
            db.session.add(suspect)
            db.session.commit()
            suspect_id = suspect.id

        real_img = cv2.imread(temp_image)

        with patch("crimai.face_engine.get_app", return_value=mock_face_engine), \
             patch("crimai.face_engine.cv2.imread", return_value=real_img):
            enroll_single(app, suspect_id)

        with app.app_context():
            updated = db.session.get(Suspect, suspect_id)
            assert updated.gallery is not None, "suspect.gallery should not be None after enrollment"

    def test_enroll_single_gallery_contains_embeddings(self, app, temp_image, mock_face_engine):
        """After enroll_single(), the gallery can be unpickled and contains at least one embedding.

        Validates: Requirements 3.6
        """
        with app.app_context():
            suspect = Suspect(
                name="Embedding Suspect",
                region="East",
                photo_path="uploads/suspects/suspect_photo.jpg",
                enroll_status="processing",
            )
            db.session.add(suspect)
            db.session.commit()
            suspect_id = suspect.id

        real_img = cv2.imread(temp_image)

        with patch("crimai.face_engine.get_app", return_value=mock_face_engine), \
             patch("crimai.face_engine.cv2.imread", return_value=real_img):
            enroll_single(app, suspect_id)

        with app.app_context():
            updated = db.session.get(Suspect, suspect_id)
            assert updated.gallery is not None

            gallery: list[np.ndarray] = pickle.loads(updated.gallery)
            assert len(gallery) >= 1, f"Expected at least one embedding, got {len(gallery)}"

            # Each entry should be a 512-dim float32 L2-normalised embedding
            for emb in gallery:
                assert isinstance(emb, np.ndarray), f"Expected np.ndarray, got {type(emb)}"
                assert emb.shape == (512,), f"Expected shape (512,), got {emb.shape}"
                assert emb.dtype == np.float32, f"Expected float32, got {emb.dtype}"
                # L2-normalised: norm should be close to 1.0
                norm = float(np.linalg.norm(emb))
                assert abs(norm - 1.0) < 1e-5, f"Embedding is not L2-normalised: norm={norm}"
