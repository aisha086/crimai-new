"""
crimai/storage.py
=================
Unified file storage abstraction.

When SUPABASE_URL and SUPABASE_SERVICE_KEY are set, all file operations
go to Supabase Storage buckets.  Otherwise files are saved to the local
crimai/static/ directory (development mode).

Public API
----------
    save_file(local_path, storage_path)  → storage_path (str)
    public_url(storage_path)             → full URL (str)
    delete_file(storage_path)            → None
    download_to_temp(storage_path)       → local temp file path (str)
    is_supabase_enabled()                → bool

Storage paths are always forward-slash relative strings like:
    "uploads/evidence/abc123.mp4"
    "uploads/suspects/face.jpg"
    "uploads/crops/det_xyz.jpg"
    "uploads/output/out_abc.mp4"

These are the same strings stored in the database.
"""

from __future__ import annotations

import logging
import os
import tempfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy Supabase client — only initialised when env vars are present
# ---------------------------------------------------------------------------
_supabase_client = None
_supabase_initialised = False


def _get_client():
    """Return the Supabase client, initialising it on first call."""
    global _supabase_client, _supabase_initialised

    if _supabase_initialised:
        return _supabase_client

    _supabase_initialised = True
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "")

    if not url or not key:
        logger.info("storage: SUPABASE_URL/SUPABASE_SERVICE_KEY not set — using local disk.")
        _supabase_client = None
        return None

    try:
        from supabase import create_client  # type: ignore
        _supabase_client = create_client(url, key)
        logger.info("storage: Supabase Storage client initialised.")
    except Exception as exc:
        logger.error("storage: Failed to initialise Supabase client: %s", exc)
        _supabase_client = None

    return _supabase_client


def is_supabase_enabled() -> bool:
    """Return True if Supabase Storage is configured and available."""
    return _get_client() is not None


# ---------------------------------------------------------------------------
# Bucket name — single bucket, organised by path prefix
# ---------------------------------------------------------------------------
BUCKET = os.environ.get("SUPABASE_BUCKET", "crimai-media")


# ---------------------------------------------------------------------------
# Local static folder (fallback)
# ---------------------------------------------------------------------------
def _local_static() -> str:
    from crimai.config import STATIC_FOLDER
    return STATIC_FOLDER


def _local_abs(storage_path: str) -> str:
    """Convert a storage_path to an absolute local path under crimai/static/."""
    return os.path.join(_local_static(), storage_path.replace("/", os.sep))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_file(local_path: str, storage_path: str) -> str:
    """Upload *local_path* to storage at *storage_path*.

    Parameters
    ----------
    local_path   : absolute path to the file on disk (already written by caller)
    storage_path : relative path to store as, e.g. "uploads/evidence/abc.mp4"

    Returns
    -------
    storage_path unchanged — callers store this in the database.
    """
    client = _get_client()

    if client is None:
        # Local mode — file is already at local_path; ensure it's in the right place
        dest = _local_abs(storage_path)
        if os.path.abspath(local_path) != os.path.abspath(dest):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            import shutil
            shutil.copy2(local_path, dest)
        return storage_path

    # Supabase mode — upload the file
    try:
        with open(local_path, "rb") as f:
            data = f.read()

        # Determine content type
        ext = os.path.splitext(local_path)[1].lower()
        content_type_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".mp4": "video/mp4", ".avi": "video/x-msvideo",
            ".mov": "video/quicktime", ".mkv": "video/x-matroska",
        }
        content_type = content_type_map.get(ext, "application/octet-stream")

        client.storage.from_(BUCKET).upload(
            path=storage_path,
            file=data,
            file_options={"content-type": content_type, "upsert": "true"},
        )
        logger.debug("storage: uploaded %s → bucket/%s", local_path, storage_path)

        # Remove the local temp file after successful upload (optional)
        # os.remove(local_path)

    except Exception as exc:
        logger.error("storage: upload failed for %s: %s", storage_path, exc)
        # Don't raise — fall back to local path so processing continues

    return storage_path


def public_url(storage_path: str) -> str:
    """Return a URL that can be used in an <img> or <video> src attribute.

    In local mode  → "/static/uploads/..."
    In Supabase mode → "https://xxx.supabase.co/storage/v1/object/public/..."
    """
    if not storage_path:
        return ""

    client = _get_client()

    if client is None:
        # Local: storage_path is relative to static root
        return "/static/" + storage_path.replace("\\", "/")

    try:
        result = client.storage.from_(BUCKET).get_public_url(storage_path)
        return result
    except Exception as exc:
        logger.error("storage: get_public_url failed for %s: %s", storage_path, exc)
        return "/static/" + storage_path.replace("\\", "/")


def delete_file(storage_path: str) -> None:
    """Delete a file from storage (best-effort, never raises)."""
    if not storage_path:
        return

    client = _get_client()

    if client is None:
        local = _local_abs(storage_path)
        try:
            if os.path.isfile(local):
                os.remove(local)
        except OSError as exc:
            logger.warning("storage: local delete failed for %s: %s", storage_path, exc)
        return

    try:
        client.storage.from_(BUCKET).remove([storage_path])
    except Exception as exc:
        logger.warning("storage: remote delete failed for %s: %s", storage_path, exc)


def download_to_temp(storage_path: str) -> str:
    """Download a file from storage to a local temp file and return its path.

    Used by background threads that need to read a file (e.g. video frames).
    The caller is responsible for deleting the temp file when done.
    """
    client = _get_client()

    if client is None:
        # Local mode — file is already on disk
        return _local_abs(storage_path)

    try:
        data = client.storage.from_(BUCKET).download(storage_path)
        ext = os.path.splitext(storage_path)[1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp.write(data)
        tmp.close()
        logger.debug("storage: downloaded %s → %s", storage_path, tmp.name)
        return tmp.name
    except Exception as exc:
        logger.error("storage: download failed for %s: %s", storage_path, exc)
        # Fall back to local path
        return _local_abs(storage_path)


def local_write_path(storage_path: str) -> str:
    """Return a local absolute path suitable for writing a new file.

    In local mode  → the final destination path (crimai/static/...)
    In Supabase mode → a temp file path; caller must call save_file() afterwards

    Always creates the parent directory.
    """
    if not is_supabase_enabled():
        dest = _local_abs(storage_path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        return dest

    # Supabase mode: write to a temp file first
    ext = os.path.splitext(storage_path)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.close()
    return tmp.name
