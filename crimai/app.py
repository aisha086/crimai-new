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
    app.config.setdefault("SQLALCHEMY_DATABASE_URI", "sqlite:///crimai.db")
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

    return app
