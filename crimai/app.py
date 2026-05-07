"""
CrimAI application factory.

Usage::

    from crimai.app import create_app
    app = create_app()

Pass a config object (e.g. a test config class) via ``config_object`` to
override the defaults loaded from ``crimai.config``.
"""

import logging
import os

from flask import Flask

from crimai.models import db, login_manager

logger = logging.getLogger(__name__)


def create_app(config_object=None) -> Flask:
    """Create and configure the Flask application.

    Steps
    -----
    1. Create the Flask instance with template/static folders.
    2. Load defaults from ``crimai.config``.
    3. Optionally override with *config_object* (useful for testing).
    4. Initialise SQLAlchemy and Flask-Login extensions.
    5. Inside an app context: create all DB tables and upload directories.
    6. Pre-warm the FaceEngine singleton (non-fatal if it fails).
    7. Register the ``auth`` and ``main`` blueprints.

    Requirements: 2.1, 2.2, 2.4, 2.5, 18.1, 18.2, 18.3, 18.4
    """
    from crimai import config as _cfg  # noqa: PLC0415

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder=_cfg.STATIC_FOLDER,   # absolute path → crimai/static/
        static_url_path="/static",
    )

    # ------------------------------------------------------------------
    # 1. Load configuration defaults from crimai.config
    # ------------------------------------------------------------------
    app.config["SECRET_KEY"] = _cfg.SECRET_KEY
    # DATABASE_URL env var → Supabase/Postgres in production
    # Falls back to local SQLite for development
    app.config.setdefault(
        "SQLALCHEMY_DATABASE_URI",
        os.environ.get("DATABASE_URL", "sqlite:///crimai.db"),
    )
    # Postgres connection pool settings (ignored by SQLite)
    # Note: sslmode=require is already embedded in the DATABASE_URL query string
    # from Supabase — no need to add it to connect_args separately.
    app.config.setdefault("SQLALCHEMY_ENGINE_OPTIONS", {
        "pool_pre_ping": True,   # test connections before use
        "pool_recycle": 300,     # recycle connections every 5 min
    })
    app.config.setdefault("UPLOAD_EVIDENCE", _cfg.UPLOAD_EVIDENCE)
    app.config.setdefault("UPLOAD_SUSPECTS", _cfg.UPLOAD_SUSPECTS)
    app.config.setdefault("UPLOAD_CROPS", _cfg.UPLOAD_CROPS)
    app.config.setdefault("UPLOAD_OUTPUT", _cfg.UPLOAD_OUTPUT)

    # ------------------------------------------------------------------
    # 2. Optional override (e.g. test config)
    # ------------------------------------------------------------------
    if config_object is not None:
        app.config.from_object(config_object)

    # ------------------------------------------------------------------
    # 3. Initialise extensions
    # ------------------------------------------------------------------
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # 4. App-context setup: tables + upload directories
    # ------------------------------------------------------------------
    with app.app_context():
        db.create_all()

        for dir_key in ("UPLOAD_EVIDENCE", "UPLOAD_SUSPECTS", "UPLOAD_CROPS", "UPLOAD_OUTPUT"):
            os.makedirs(app.config[dir_key], exist_ok=True)

        # --------------------------------------------------------------
        # 5. Pre-warm FaceEngine (non-fatal)
        # --------------------------------------------------------------
        try:
            from crimai.face_engine import get_app as _get_face_app  # noqa: PLC0415

            _get_face_app()
        except Exception as exc:  # noqa: BLE001
            logger.error("FaceEngine pre-warm failed — face recognition unavailable: %s", exc)

    # ------------------------------------------------------------------
    # 6. Register blueprints (lazy imports to avoid circular imports)
    # ------------------------------------------------------------------
    from crimai.auth import auth as auth_blueprint  # noqa: PLC0415
    from crimai.main import main as main_blueprint  # noqa: PLC0415

    app.register_blueprint(auth_blueprint, url_prefix="/auth")
    app.register_blueprint(main_blueprint, url_prefix="/")

    # Make storage_url available in all templates
    from crimai import storage as _storage  # noqa: PLC0415

    @app.template_global()
    def storage_url(path: str) -> str:
        """Return the correct URL for a stored file (local or Supabase)."""
        return _storage.public_url(path)

    return app
