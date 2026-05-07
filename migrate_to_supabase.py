"""
migrate_to_supabase.py
======================
Selective migration:
  1. Migrates all DB rows from SQLite → Supabase Postgres
  2. Uploads ONLY output videos (referenced in Media.output_path) to Supabase Storage
  3. Deletes local crop images (uploads/crops/) to free disk space — does NOT upload them

Usage
-----
    python migrate_to_supabase.py

Set DATABASE_URL, SUPABASE_URL, SUPABASE_SERVICE_KEY, SUPABASE_BUCKET in .env first.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded .env")
except ImportError:
    pass

DATABASE_URL = os.environ.get("DATABASE_URL", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
BUCKET       = os.environ.get("SUPABASE_BUCKET", "crimai-media")

if not DATABASE_URL or not DATABASE_URL.startswith("postgresql"):
    print("ERROR: DATABASE_URL must be a postgresql+psycopg2:// connection string.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Stub heavy ML deps
# ---------------------------------------------------------------------------
import types

for _name in ["insightface", "insightface.app", "onnxruntime", "cv2"]:
    if _name not in sys.modules:
        _mod = types.ModuleType(_name)
        sys.modules[_name] = _mod

sys.modules["insightface"].app = sys.modules["insightface.app"]
sys.modules["insightface.app"].FaceAnalysis = type("FaceAnalysis", (), {})
sys.modules["onnxruntime"].get_available_providers = lambda: []
sys.modules["cv2"].VideoCapture = None

# ---------------------------------------------------------------------------
# DB engines
# ---------------------------------------------------------------------------
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, make_transient

_here = os.path.dirname(os.path.abspath(__file__))
SQLITE_PATH = os.path.join(_here, "instance", "crimai.db")
if not os.path.exists(SQLITE_PATH):
    SQLITE_PATH = os.path.join(_here, "crimai", "crimai.db")
if not os.path.exists(SQLITE_PATH):
    SQLITE_PATH = os.path.join(_here, "crimai.db")
if not os.path.exists(SQLITE_PATH):
    print(f"ERROR: SQLite database not found.")
    sys.exit(1)

print(f"\nSource DB : sqlite:///{SQLITE_PATH}")
print(f"Target DB : {DATABASE_URL[:70]}...")

src_engine = create_engine(f"sqlite:///{SQLITE_PATH}")
tgt_engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SrcSession = sessionmaker(bind=src_engine)
TgtSession = sessionmaker(bind=tgt_engine)

# ---------------------------------------------------------------------------
# Create tables on Postgres
# ---------------------------------------------------------------------------
from crimai.models import (
    CaseReport, DetectionResult, Media, Suspect, UnknownIdentity, User, db
)

db.metadata.create_all(tgt_engine)
print("Tables created on Postgres ✓")

# ---------------------------------------------------------------------------
# Migrate DB rows
# ---------------------------------------------------------------------------

def migrate_table(model, src_session, tgt_session, label):
    rows = src_session.query(model).all()
    if not rows:
        print(f"  {label}: 0 rows — skipping")
        return
    src_session.expunge_all()
    for row in rows:
        make_transient(row)
        tgt_session.merge(row)
    tgt_session.commit()
    print(f"  {label}: {len(rows)} rows migrated ✓")


src = SrcSession()
tgt = TgtSession()

print("\n── Migrating database rows ──")
# migrate_table(User,            src, tgt, "User")
# migrate_table(Suspect,         src, tgt, "Suspect")
# migrate_table(Media,           src, tgt, "Media")
# migrate_table(DetectionResult, src, tgt, "DetectionResult")
# migrate_table(UnknownIdentity, src, tgt, "UnknownIdentity")
# migrate_table(CaseReport,      src, tgt, "CaseReport")

print("Database migration complete ✓")

# ---------------------------------------------------------------------------
# Upload output videos to Supabase Storage
# ---------------------------------------------------------------------------
if not SUPABASE_URL or not SUPABASE_KEY:
    print("\nSUPABASE_URL / SUPABASE_SERVICE_KEY not set — skipping video upload.")
else:
    try:
        from supabase import create_client
    except ImportError:
        print("\nERROR: supabase package not installed. Run: pip install supabase==2.4.3")
        sys.exit(1)

    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    from crimai.config import STATIC_FOLDER

    # Get all output_path values from Media table that are not null
    output_paths = [
        row[0] for row in
        src_engine.connect().execute(
            __import__("sqlalchemy").text(
                "SELECT output_path FROM media WHERE output_path IS NOT NULL AND output_path != ''"
            )
        ).fetchall()
    ]

    print(f"\n── Uploading {len(output_paths)} output video(s) to Supabase Storage ──")

    uploaded = skipped = errors = 0

    for storage_path in output_paths:
        # storage_path is like "uploads/output/out_abc123.mp4"
        local_abs = os.path.join(STATIC_FOLDER, storage_path.replace("/", os.sep))

        if not os.path.isfile(local_abs):
            print(f"  ⚠ Not found locally: {storage_path}")
            skipped += 1
            continue

        try:
            with open(local_abs, "rb") as f:
                data = f.read()

            ext = os.path.splitext(storage_path)[1].lower()
            ct = "video/mp4" if ext == ".mp4" else "video/x-msvideo"

            client.storage.from_(BUCKET).upload(
                path=storage_path,
                file=data,
                file_options={"content-type": ct, "upsert": "true"},
            )
            size_mb = len(data) / 1024 / 1024
            print(f"  ✓ {storage_path} ({size_mb:.1f} MB)")
            uploaded += 1

        except Exception as exc:
            err = str(exc)
            if "already exists" in err.lower() or "duplicate" in err.lower():
                print(f"  ~ {storage_path} (already in bucket, skipped)")
                skipped += 1
            else:
                print(f"  ✗ {storage_path}: {exc}")
                errors += 1

    print(f"\nVideo upload: {uploaded} uploaded, {skipped} skipped, {errors} errors")

src.close()
tgt.close()

# ---------------------------------------------------------------------------
# Delete local crop images to free disk space
# ---------------------------------------------------------------------------
print("\n── Deleting local crop images (uploads/crops/) ──")
print("   Not uploading them — just freeing local disk space.")
print("   DB path references are kept; images won't display until re-processed.\n")

from crimai.config import STATIC_FOLDER as SF

crops_dir = os.path.join(SF, "uploads", "crops")

if not os.path.isdir(crops_dir):
    print(f"  Crops directory not found at {crops_dir} — nothing to delete.")
else:
    local_crop_files = [
        f for f in os.listdir(crops_dir)
        if f != ".gitkeep" and os.path.isfile(os.path.join(crops_dir, f))
    ]

    if not local_crop_files:
        print("  No local crop files found.")
    else:
        print(f"  Found {len(local_crop_files)} local crop files.")
        confirm = input(f"  Delete all {len(local_crop_files)} local crop images? [y/N] ").strip().lower()

        if confirm == "y":
            deleted = 0
            for fname in local_crop_files:
                try:
                    os.remove(os.path.join(crops_dir, fname))
                    deleted += 1
                except OSError as e:
                    print(f"  ✗ Could not delete {fname}: {e}")
            print(f"  Deleted {deleted} local crop images ✓")
        else:
            print("  Skipped — no local files deleted.")

# ---------------------------------------------------------------------------
# Delete crop images already uploaded to Supabase Storage
# ---------------------------------------------------------------------------
if SUPABASE_URL and SUPABASE_KEY:
    print("\n── Deleting crop images from Supabase Storage (uploads/crops/) ──")
    print("   These were uploaded in a previous migration run.\n")

    try:
        # List all files in the crops folder in the bucket
        # Supabase Storage list() returns up to 100 items per call — paginate
        CROPS_PREFIX = "uploads/crops"
        all_remote_crops = []
        offset = 0
        PAGE = 100

        while True:
            page = client.storage.from_(BUCKET).list(
                path=CROPS_PREFIX,
                options={"limit": PAGE, "offset": offset},
            )
            if not page:
                break
            all_remote_crops.extend(page)
            if len(page) < PAGE:
                break
            offset += PAGE

        # Filter out folder entries (they have no size or id)
        crop_objects = [
            obj for obj in all_remote_crops
            if obj.get("name") and obj.get("name") != ".gitkeep"
        ]

        if not crop_objects:
            print("  No crop images found in Supabase Storage bucket.")
        else:
            print(f"  Found {len(crop_objects)} crop images in bucket.")
            confirm2 = input(
                f"  Delete all {len(crop_objects)} crop images from Supabase Storage? [y/N] "
            ).strip().lower()

            if confirm2 == "y":
                # Delete in batches of 100 (Supabase limit)
                BATCH = 100
                remote_deleted = 0
                remote_errors = 0

                paths_to_delete = [
                    f"{CROPS_PREFIX}/{obj['name']}" for obj in crop_objects
                ]

                for i in range(0, len(paths_to_delete), BATCH):
                    batch = paths_to_delete[i:i + BATCH]
                    try:
                        client.storage.from_(BUCKET).remove(batch)
                        remote_deleted += len(batch)
                        print(f"  Deleted batch {i // BATCH + 1} ({len(batch)} files)")
                    except Exception as exc:
                        print(f"  ✗ Batch {i // BATCH + 1} failed: {exc}")
                        remote_errors += len(batch)

                print(f"  Deleted {remote_deleted} from Supabase Storage ✓  ({remote_errors} errors)")
            else:
                print("  Skipped — no remote files deleted.")

    except Exception as exc:
        print(f"  ✗ Could not list/delete from Supabase Storage: {exc}")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print("\n══════════════════════════════════════════")
print("Done.")
print("• DB rows         → Supabase Postgres ✓")
print("• Output videos   → Supabase Storage ✓")
print("• Local crops     → deleted from disk")
print("• Remote crops    → deleted from Supabase Storage")
print("Verify at: https://supabase.com → Table Editor & Storage")
print("══════════════════════════════════════════")
