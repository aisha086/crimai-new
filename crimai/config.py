"""
CrimAI application configuration constants.
All modules should import constants exclusively from this file.
"""

import os

# Root of the crimai package — used to build absolute paths so that
# file saves and Flask's static serving always point to the same directory.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Face recognition model settings
# ---------------------------------------------------------------------------
MODEL_NAME = "buffalo_l"
DET_SIZE_CPU = (640, 640)
DET_SIZE_GPU = (1024, 1024)
DET_THRESH = 0.35
SIMILARITY_THRESH = 0.50
FRAME_SKIP = 3
ALERT_DURATION_FRAMES = 45

# ---------------------------------------------------------------------------
# File storage paths — absolute, rooted at crimai/static/
# ---------------------------------------------------------------------------
STATIC_FOLDER   = os.path.join(_PKG_DIR, "static")
UPLOAD_EVIDENCE = os.path.join(STATIC_FOLDER, "uploads", "evidence")
UPLOAD_SUSPECTS = os.path.join(STATIC_FOLDER, "uploads", "suspects")
UPLOAD_CROPS    = os.path.join(STATIC_FOLDER, "uploads", "crops")
UPLOAD_OUTPUT   = os.path.join(STATIC_FOLDER, "uploads", "output")

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
SECRET_KEY = os.environ.get("SECRET_KEY") or "crimai-dev-secret-key-change-in-production"

# ---------------------------------------------------------------------------
# Allowed upload extensions
# ---------------------------------------------------------------------------
ALLOWED_IMAGE_EXT = {"jpg", "jpeg", "png"}
ALLOWED_VIDEO_EXT = {"mp4", "avi", "mov", "mkv"}

# ---------------------------------------------------------------------------
# Geographic regions
# ---------------------------------------------------------------------------
REGIONS = ["North", "South", "East", "West", "Central", "Unknown"]
