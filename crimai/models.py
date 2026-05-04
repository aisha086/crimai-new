"""
CrimAI ORM models.

All six SQLAlchemy models are defined here, along with the shared `db`
(SQLAlchemy) and `login_manager` (LoginManager) extension instances that are
initialised against the Flask app inside `create_app()`.
"""

from datetime import datetime

from flask_login import LoginManager, UserMixin
from flask_sqlalchemy import SQLAlchemy

# ---------------------------------------------------------------------------
# Extension singletons — initialised via db.init_app() / login_manager.init_app()
# inside the application factory.
# ---------------------------------------------------------------------------
db = SQLAlchemy()
login_manager = LoginManager()


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------
class User(UserMixin, db.Model):
    """Authenticated operator account.

    Implements Flask-Login's UserMixin so that `current_user` works out of the
    box.  Passwords are stored as PBKDF2-SHA256 hashes — never as plaintext
    (Requirements 1.2, 1.8).
    """

    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<User {self.username!r}>"


@login_manager.user_loader
def load_user(user_id: str) -> "User | None":
    """Flask-Login callback: load a User by primary key."""
    return db.session.get(User, int(user_id))


# ---------------------------------------------------------------------------
# Suspect
# ---------------------------------------------------------------------------
class Suspect(db.Model):
    """Watchlist entry with face gallery.

    The `gallery` column stores a ``pickle.dumps(list[np.ndarray])`` byte
    string (Requirements 16.1, 16.2).  Cascade delete propagates to all
    associated DetectionResult rows (Requirement 17.1).
    """

    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(120), nullable=False)
    region        = db.Column(db.String(80))
    photo_path    = db.Column(db.String(512))
    gallery       = db.Column(db.LargeBinary)          # pickle.dumps(list[np.ndarray])
    enroll_mode   = db.Column(db.String(20), default='single')
    enroll_status = db.Column(db.String(20), default='processing')
    error_msg     = db.Column(db.Text)
    added_at      = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    detections    = db.relationship(
        'DetectionResult',
        backref='suspect',
        cascade='all, delete-orphan',
    )

    def __repr__(self) -> str:
        return f"<Suspect {self.id} {self.name!r} [{self.enroll_status}]>"


# ---------------------------------------------------------------------------
# Media
# ---------------------------------------------------------------------------
class Media(db.Model):
    """Uploaded evidence file (image or video).

    Cascade deletes propagate to DetectionResult, UnknownIdentity, and
    CaseReport rows (Requirement 17.2).  The one-to-one relationship with
    CaseReport is enforced via ``uselist=False`` at the ORM layer and
    ``unique=True`` on ``CaseReport.media_id`` at the schema layer
    (Requirement 17.3).
    """

    id            = db.Column(db.Integer, primary_key=True)
    filename      = db.Column(db.String(256))
    file_path     = db.Column(db.String(512))
    file_type     = db.Column(db.String(10))           # 'image' | 'video'
    case_id       = db.Column(db.String(80))
    priority      = db.Column(db.String(20), default='Medium')
    description   = db.Column(db.Text)
    location      = db.Column(db.String(256))
    region        = db.Column(db.String(80))
    inspector     = db.Column(db.String(120))
    upload_mode   = db.Column(db.String(20))
    status        = db.Column(db.String(20), default='pending')
    progress      = db.Column(db.Integer, default=0)
    output_path   = db.Column(db.String(512))
    error_msg     = db.Column(db.Text)
    uploaded_at   = db.Column(db.DateTime, default=datetime.utcnow)
    finished_at   = db.Column(db.DateTime)
    processed     = db.Column(db.Boolean, default=False)

    # Relationships
    detections    = db.relationship(
        'DetectionResult',
        backref='media',
        cascade='all, delete-orphan',
    )
    unknowns      = db.relationship(
        'UnknownIdentity',
        backref='media',
        cascade='all, delete-orphan',
    )
    report        = db.relationship(
        'CaseReport',
        backref='media',
        cascade='all, delete-orphan',
        uselist=False,
    )

    def __repr__(self) -> str:
        return f"<Media {self.id} {self.filename!r} [{self.status}]>"


# ---------------------------------------------------------------------------
# DetectionResult
# ---------------------------------------------------------------------------
class DetectionResult(db.Model):
    """A single confirmed face-match between a Media file and a Suspect.

    Foreign keys carry ``ondelete='CASCADE'`` so that the database enforces
    referential integrity even when rows are deleted outside the ORM
    (Requirements 17.1, 17.2).
    """

    id            = db.Column(db.Integer, primary_key=True)
    suspect_id    = db.Column(
        db.Integer,
        db.ForeignKey('suspect.id', ondelete='CASCADE'),
    )
    media_id      = db.Column(
        db.Integer,
        db.ForeignKey('media.id', ondelete='CASCADE'),
    )
    confidence    = db.Column(db.Float)
    frame_number  = db.Column(db.Integer)
    timestamp_sec = db.Column(db.Float)
    crop_path     = db.Column(db.String(512))
    frame_path    = db.Column(db.String(512))
    detected_at   = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return (
            f"<DetectionResult {self.id} suspect={self.suspect_id} "
            f"media={self.media_id} conf={self.confidence:.3f}>"
        )


# ---------------------------------------------------------------------------
# UnknownIdentity
# ---------------------------------------------------------------------------
class UnknownIdentity(db.Model):
    """An unmatched face extracted from a Media file.

    The `embedding` column stores a ``pickle.dumps(np.ndarray)`` byte string
    (Requirements 16.4, 16.5).
    """

    id              = db.Column(db.Integer, primary_key=True)
    media_id        = db.Column(
        db.Integer,
        db.ForeignKey('media.id', ondelete='CASCADE'),
    )
    crop_path       = db.Column(db.String(512))
    embedding       = db.Column(db.LargeBinary)        # pickle.dumps(np.ndarray)
    best_score      = db.Column(db.Float)
    closest_suspect = db.Column(db.String(120))
    detected_at     = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return (
            f"<UnknownIdentity {self.id} media={self.media_id} "
            f"best_score={self.best_score}>"
        )


# ---------------------------------------------------------------------------
# CaseReport
# ---------------------------------------------------------------------------
class CaseReport(db.Model):
    """Rich-text HTML report associated with a single Media record.

    ``unique=True`` on ``media_id`` enforces the one-to-one constraint at the
    database schema level (Requirement 17.3).  The ``onupdate`` hook on
    ``updated_at`` keeps the timestamp current on every ORM flush.
    """

    id           = db.Column(db.Integer, primary_key=True)
    media_id     = db.Column(
        db.Integer,
        db.ForeignKey('media.id', ondelete='CASCADE'),
        unique=True,
    )
    html_content = db.Column(db.Text)
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at   = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    def __repr__(self) -> str:
        return f"<CaseReport {self.id} media={self.media_id}>"
