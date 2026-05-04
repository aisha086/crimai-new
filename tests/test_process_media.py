"""
Integration test for process_media() — video branch.

insightface and onnxruntime are heavy ML dependencies not available in CI.
We stub them in sys.modules before importing face_engine so that the module
loads without errors.

Validates: Requirements 7.11
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

import os
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from flask import Flask

from crimai.face_engine import process_media
from crimai.models import DetectionResult, Media, UnknownIdentity, db, login_manager

# Save the real cv2.VideoWriter before any patching so we can use it in helpers.
_RealVideoWriter = cv2.VideoWriter


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
def synthetic_video(tmp_path):
    """Create a short synthetic AVI video (8 frames, 320x240) and return its path."""
    video_path = str(tmp_path / "test_video.avi")
    fourcc = _RealVideoWriter.fourcc(*"XVID")
    writer = _RealVideoWriter(video_path, fourcc, 10.0, (320, 240))

    for i in range(8):
        # Create a simple BGR frame with a gradient so frames are non-trivial
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:, :, 0] = i * 30          # blue channel varies per frame
        frame[50:150, 80:200] = [200, 180, 160]  # face-like rectangle
        writer.write(frame)

    writer.release()
    return video_path


@pytest.fixture
def fake_face():
    """Return a mock face object with a 512-dim embedding and a bbox."""
    face = MagicMock()
    face.embedding = np.ones(512, dtype=np.float32)
    face.bbox = [10.0, 10.0, 100.0, 100.0]
    return face


@pytest.fixture
def mock_face_engine(fake_face):
    """Return a mock FaceAnalysis object that always returns one fake face."""
    fa = MagicMock()
    fa.det_thresh = 0.35
    fa.get.return_value = [fake_face]
    return fa


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_mock_cap(synthetic_video_path: str):
    """Build a mock cv2.VideoCapture that replays frames from a real video file."""
    real_cap = cv2.VideoCapture(synthetic_video_path)

    # Collect all frames up front so we can replay them deterministically.
    frames = []
    while True:
        ret, frame = real_cap.read()
        if not ret:
            break
        frames.append(frame)

    fps = real_cap.get(cv2.CAP_PROP_FPS) or 10.0
    width = real_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 320.0
    height = real_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 240.0
    real_cap.release()

    frame_iter = iter(frames)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True

    prop_map = {
        cv2.CAP_PROP_FPS: fps,
        cv2.CAP_PROP_FRAME_WIDTH: width,
        cv2.CAP_PROP_FRAME_HEIGHT: height,
        cv2.CAP_PROP_FRAME_COUNT: float(len(frames)),
    }
    mock_cap.get.side_effect = lambda prop: prop_map.get(prop, 0.0)

    def _read():
        try:
            return True, next(frame_iter)
        except StopIteration:
            return False, None

    mock_cap.read.side_effect = _read
    mock_cap.release.return_value = None

    return mock_cap, len(frames)


# ---------------------------------------------------------------------------
# Integration test — Task 7.3
# Validates: Requirements 7.11
# ---------------------------------------------------------------------------

class TestProcessMediaVideo:
    """Integration tests for process_media() on a synthetic video."""

    def test_process_media_video_status_done(
        self, app, synthetic_video, mock_face_engine, tmp_path
    ):
        """After process_media() on a synthetic video, media.status must be 'done'.

        Validates: Requirements 7.11
        """
        # Create a Media record pointing to the synthetic video.
        with app.app_context():
            media = Media(
                filename="test_video.avi",
                file_path="uploads/evidence/test_video.avi",
                file_type="video",
                case_id="TEST-001",
                status="pending",
            )
            db.session.add(media)
            db.session.commit()
            media_id = media.id

        mock_cap, _ = _build_mock_cap(synthetic_video)
        avi_out = str(tmp_path / "out.avi")

        # Use the real VideoWriter (saved before patching) to avoid recursion.
        def _make_writer(path, fourcc, fps, size):
            return _RealVideoWriter(avi_out, fourcc, fps, size)

        with (
            patch("crimai.face_engine.get_app", return_value=mock_face_engine),
            patch("crimai.face_engine.subprocess.run", return_value=MagicMock(returncode=1, stderr=b"ffmpeg not found")),
            patch("crimai.face_engine.cv2.VideoCapture", return_value=mock_cap),
            patch("crimai.face_engine.cv2.VideoWriter", side_effect=_make_writer),
            patch("crimai.face_engine.cv2.imwrite", return_value=True),
            patch("os.makedirs"),
        ):
            process_media(app, media_id)

        with app.app_context():
            updated = db.session.get(Media, media_id)
            assert updated.status == "done", (
                f"Expected media.status='done', got {updated.status!r}. "
                f"error_msg={updated.error_msg!r}"
            )

    def test_process_media_video_creates_detection_or_unknown(
        self, app, synthetic_video, mock_face_engine, tmp_path
    ):
        """After process_media() on a synthetic video, at least one DetectionResult
        or UnknownIdentity record must exist for the media.

        Validates: Requirements 7.11
        """
        with app.app_context():
            media = Media(
                filename="test_video2.avi",
                file_path="uploads/evidence/test_video2.avi",
                file_type="video",
                case_id="TEST-002",
                status="pending",
            )
            db.session.add(media)
            db.session.commit()
            media_id = media.id

        mock_cap, _ = _build_mock_cap(synthetic_video)
        avi_out = str(tmp_path / "out2.avi")

        def _make_writer(path, fourcc, fps, size):
            return _RealVideoWriter(avi_out, fourcc, fps, size)

        with (
            patch("crimai.face_engine.get_app", return_value=mock_face_engine),
            patch("crimai.face_engine.subprocess.run", return_value=MagicMock(returncode=1, stderr=b"ffmpeg not found")),
            patch("crimai.face_engine.cv2.VideoCapture", return_value=mock_cap),
            patch("crimai.face_engine.cv2.VideoWriter", side_effect=_make_writer),
            patch("crimai.face_engine.cv2.imwrite", return_value=True),
            patch("os.makedirs"),
        ):
            process_media(app, media_id)

        with app.app_context():
            detection_count = DetectionResult.query.filter_by(media_id=media_id).count()
            unknown_count = UnknownIdentity.query.filter_by(media_id=media_id).count()
            total = detection_count + unknown_count
            assert total >= 1, (
                f"Expected at least one DetectionResult or UnknownIdentity for media {media_id}, "
                f"got detections={detection_count}, unknowns={unknown_count}"
            )
