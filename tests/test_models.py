"""
Unit tests for ORM cascade behaviour.

Validates Requirements 17.1, 17.2, 17.3:
  17.1 - Deleting a Suspect cascade-deletes its DetectionResult records.
  17.2 - Deleting a Media cascade-deletes its DetectionResult, UnknownIdentity,
         and CaseReport records.
  17.3 - One-to-one constraint between Media and CaseReport is enforced.
"""

import pytest
from flask import Flask
from sqlalchemy.exc import IntegrityError

from crimai.models import (
    CaseReport,
    DetectionResult,
    Media,
    Suspect,
    UnknownIdentity,
    db,
    login_manager,
)


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
def session(app):
    """Return the SQLAlchemy session bound to the test app context."""
    return db.session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_suspect(name="Alice") -> Suspect:
    return Suspect(name=name, region="North")


def _make_media(filename="evidence.mp4") -> Media:
    return Media(filename=filename, file_type="video", status="pending")


# ---------------------------------------------------------------------------
# Requirement 17.1 — Suspect cascade-deletes DetectionResult
# ---------------------------------------------------------------------------

class TestSuspectCascade:
    """Deleting a Suspect must remove all its DetectionResult rows (Req 17.1)."""

    def test_delete_suspect_removes_detection_results(self, session):
        suspect = _make_suspect()
        media = _make_media()
        session.add_all([suspect, media])
        session.flush()

        detection = DetectionResult(
            suspect_id=suspect.id,
            media_id=media.id,
            confidence=0.92,
        )
        session.add(detection)
        session.commit()

        detection_id = detection.id

        # Sanity check — record exists before deletion
        assert session.get(DetectionResult, detection_id) is not None

        session.delete(suspect)
        session.commit()

        # DetectionResult must be gone after Suspect is deleted
        assert session.get(DetectionResult, detection_id) is None

    def test_delete_suspect_does_not_remove_unrelated_detection(self, session):
        """Cascade must not affect DetectionResults linked to other Suspects."""
        suspect_a = _make_suspect("Alice")
        suspect_b = _make_suspect("Bob")
        media = _make_media()
        session.add_all([suspect_a, suspect_b, media])
        session.flush()

        det_a = DetectionResult(suspect_id=suspect_a.id, media_id=media.id, confidence=0.8)
        det_b = DetectionResult(suspect_id=suspect_b.id, media_id=media.id, confidence=0.7)
        session.add_all([det_a, det_b])
        session.commit()

        session.delete(suspect_a)
        session.commit()

        assert session.get(DetectionResult, det_a.id) is None
        assert session.get(DetectionResult, det_b.id) is not None


# ---------------------------------------------------------------------------
# Requirement 17.2 — Media cascade-deletes DetectionResult, UnknownIdentity,
#                    and CaseReport
# ---------------------------------------------------------------------------

class TestMediaCascade:
    """Deleting a Media must remove all child rows (Req 17.2)."""

    def test_delete_media_removes_all_children(self, session):
        suspect = _make_suspect()
        media = _make_media()
        session.add_all([suspect, media])
        session.flush()

        detection = DetectionResult(
            suspect_id=suspect.id,
            media_id=media.id,
            confidence=0.85,
        )
        unknown = UnknownIdentity(media_id=media.id, best_score=0.3)
        report = CaseReport(media_id=media.id, html_content="<p>report</p>")
        session.add_all([detection, unknown, report])
        session.commit()

        det_id = detection.id
        unk_id = unknown.id
        rep_id = report.id

        session.delete(media)
        session.commit()

        assert session.get(DetectionResult, det_id) is None
        assert session.get(UnknownIdentity, unk_id) is None
        assert session.get(CaseReport, rep_id) is None

    def test_delete_media_does_not_remove_suspect(self, session):
        """Cascade must not touch the Suspect record itself."""
        suspect = _make_suspect()
        media = _make_media()
        session.add_all([suspect, media])
        session.flush()

        detection = DetectionResult(
            suspect_id=suspect.id,
            media_id=media.id,
            confidence=0.9,
        )
        session.add(detection)
        session.commit()

        suspect_id = suspect.id

        session.delete(media)
        session.commit()

        assert session.get(Suspect, suspect_id) is not None


# ---------------------------------------------------------------------------
# Requirement 17.3 — One-to-one Media ↔ CaseReport constraint
# ---------------------------------------------------------------------------

class TestMediaCaseReportOneToOne:
    """A Media record may have at most one CaseReport (Req 17.3)."""

    def test_single_case_report_allowed(self, session):
        media = _make_media()
        session.add(media)
        session.flush()

        report = CaseReport(media_id=media.id, html_content="<p>ok</p>")
        session.add(report)
        session.commit()

        assert session.get(CaseReport, report.id) is not None

    def test_duplicate_case_report_raises_integrity_error(self, session):
        media = _make_media()
        session.add(media)
        session.flush()

        report1 = CaseReport(media_id=media.id, html_content="<p>first</p>")
        session.add(report1)
        session.commit()

        # Attempt to insert a second CaseReport for the same Media
        report2 = CaseReport(media_id=media.id, html_content="<p>second</p>")
        session.add(report2)

        with pytest.raises(IntegrityError):
            session.commit()
