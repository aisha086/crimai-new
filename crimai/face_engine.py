"""
CrimAI face recognition engine.

Provides a module-level FaceAnalysis singleton and helper functions for
suspect enrollment and evidence processing.
"""

from __future__ import annotations

import logging
import os
import pickle
import subprocess
import traceback
import uuid
from datetime import datetime

import cv2
import numpy as np
import onnxruntime
from insightface.app import FaceAnalysis

from crimai.config import (
    ALERT_DURATION_FRAMES,
    DET_SIZE_CPU,
    DET_SIZE_GPU,
    DET_THRESH,
    FRAME_SKIP,
    MODEL_NAME,
    SIMILARITY_THRESH,
    STATIC_FOLDER,
    UPLOAD_CROPS,
    UPLOAD_OUTPUT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_fa: FaceAnalysis | None = None


def get_app() -> FaceAnalysis:
    """Return the cached FaceAnalysis singleton; initialise on first call.

    Detects CUDA availability via onnxruntime and selects the appropriate
    execution provider and detection size accordingly.
    """
    global _fa

    if _fa is not None:
        return _fa

    available_providers = onnxruntime.get_available_providers()

    if "CUDAExecutionProvider" in available_providers:
        provider = "CUDAExecutionProvider"
        det_size = DET_SIZE_GPU
    else:
        provider = "CPUExecutionProvider"
        det_size = DET_SIZE_CPU

    fa = FaceAnalysis(name=MODEL_NAME, providers=[provider])
    fa.prepare(ctx_id=0, det_thresh=DET_THRESH, det_size=det_size)

    _fa = fa
    return _fa


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Apply CLAHE to BGR frame; return enhanced BGR frame.

    Converts to LAB colour space, applies CLAHE (clipLimit=3.0,
    tileGridSize=(8,8)) to the L channel, then converts back to BGR.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def get_face_embedding(
    fa, img_bgr: np.ndarray
) -> tuple[np.ndarray | None, object | None]:
    """Detect largest face; return (L2-normalised embedding, face object) or (None, None).

    Strategy:
    1. Try preprocessed frame first, then raw frame.
    2. For each frame variant, try det_thresh values: 0.35 → 0.20 → 0.10.
    3. Among all detected faces, pick the one with the largest bounding box area.
    4. Return the L2-normalised embedding of that face, or (None, None) if no face found.

    L2 normalisation: emb / np.linalg.norm(emb)
    """
    thresholds = [0.35, 0.20, 0.10]
    preprocessed = preprocess_frame(img_bgr)
    frame_variants = [preprocessed, img_bgr]

    best_face = None
    best_area = -1.0

    original_thresh = fa.det_thresh

    try:
        for frame in frame_variants:
            for thresh in thresholds:
                fa.det_thresh = thresh
                faces = fa.get(frame)
                if not faces:
                    continue
                for face in faces:
                    x1, y1, x2, y2 = face.bbox
                    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                    if area > best_area:
                        best_area = area
                        best_face = face
                if best_face is not None:
                    # Found at least one face at this threshold; no need to
                    # lower the threshold further for this frame variant.
                    break
            if best_face is not None:
                # Found a face in the preprocessed frame; skip the raw frame.
                break
    finally:
        fa.det_thresh = original_thresh

    if best_face is None:
        return None, None

    emb = best_face.embedding.astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm == 0.0:
        return None, None
    emb_normalised = emb / norm
    return emb_normalised, best_face


def load_all_embeddings() -> dict[str, list[np.ndarray]]:
    """Return {suspect_label: [embedding, ...]} for all ready Suspects.

    Queries all Suspect records with enroll_status='ready'.
    Unpickles each gallery (pickle.loads(suspect.gallery)).
    Returns dict mapping suspect label (f"{suspect.id}:{suspect.name}") to list of embeddings.
    Skips suspects with None gallery or unpickling errors (log warning).
    """
    # Import inside function body to avoid circular imports at module level.
    from crimai.models import Suspect

    result: dict[str, list[np.ndarray]] = {}

    suspects = Suspect.query.filter_by(enroll_status="ready").all()
    for suspect in suspects:
        if suspect.gallery is None:
            logger.warning(
                "Suspect %d (%r) has enroll_status='ready' but gallery is None; skipping.",
                suspect.id,
                suspect.name,
            )
            continue
        try:
            embeddings: list[np.ndarray] = pickle.loads(suspect.gallery)
        except Exception as exc:
            logger.warning(
                "Failed to unpickle gallery for suspect %d (%r): %s; skipping.",
                suspect.id,
                suspect.name,
                exc,
            )
            continue
        label = f"{suspect.id}:{suspect.name}"
        result[label] = embeddings

    return result


def match_embedding(
    emb: np.ndarray,
    suspect_embeddings: dict[str, list[np.ndarray]],
) -> tuple[str | None, float]:
    """Return (best_label, best_score) or (None, best_score) against all galleries.

    For each suspect label and their list of embeddings:
      - Compute cosine similarity: np.dot(emb, gallery_emb) for each L2-normalised embedding
      - Track the maximum score across all suspects and all their embeddings

    If best_score >= SIMILARITY_THRESH: return (best_label, best_score)
    Else: return (None, best_score)

    If suspect_embeddings is empty: return (None, 0.0)
    """
    if not suspect_embeddings:
        return None, 0.0

    best_label: str | None = None
    best_score: float = -float("inf")

    for label, embeddings in suspect_embeddings.items():
        for gallery_emb in embeddings:
            score = float(np.dot(emb, gallery_emb))
            if score > best_score:
                best_score = score
                best_label = label

    if best_score >= SIMILARITY_THRESH:
        return best_label, best_score
    return None, best_score


def _to_relative(abs_path: str) -> str:
    """Strip the static folder prefix from a path; return the part after static/.

    The returned value is suitable for use with url_for('static', filename=...).
    Works with both absolute paths and paths already relative to static/.
    """
    # Normalise to forward slashes
    normalised = os.path.normpath(abs_path).replace(os.sep, "/")
    static_norm = os.path.normpath(STATIC_FOLDER).replace(os.sep, "/")

    # Case 1: absolute path starting with the static folder
    if normalised.startswith(static_norm + "/"):
        return normalised[len(static_norm) + 1:]

    # Case 2: path contains "static/" somewhere (legacy relative paths)
    marker = "static/"
    idx = normalised.find(marker)
    if idx != -1:
        return normalised[idx + len(marker):]

    # Case 3: already relative — return as-is
    return normalised


def enroll_single(app, suspect_id: int) -> None:
    """Background thread target: build Gallery for a single-photo Suspect.

    1. Push an app context
    2. Load the Suspect record by suspect_id
    3. Read the photo from suspect.photo_path (which is relative to static folder)
    4. Call get_face_embedding() on the photo to get the base embedding
    5. Generate augmented variants:
       - 4 brightness/contrast adjustments: alpha in [0.8, 0.9, 1.1, 1.2], beta in [-20, -10, 10, 20]
       - 2 Gaussian blur variants: kernel sizes (3,3) and (5,5)
    6. Extract embeddings from each augmented variant where a face is detected
    7. Collect all valid embeddings into a gallery list
    8. If no embeddings found: set enroll_status='failed', error_msg='No face detected'
    9. Else: pickle the gallery list, write to suspect.gallery, set enroll_status='ready'
    10. On any exception: set enroll_status='failed', error_msg=str(e), log ERROR with traceback

    The photo path in suspect.photo_path is relative to the static folder.
    Reconstruct the full path as: os.path.join(STATIC_FOLDER, suspect.photo_path)
    """
    # Import inside function body to avoid circular imports at module level.
    from crimai.models import Suspect, db

    with app.app_context():
        suspect = db.session.get(Suspect, suspect_id)
        if suspect is None:
            logger.error(
                "enroll_single: Suspect %d not found in database.", suspect_id
            )
            return

        try:
            # Reconstruct the full path to the photo
            full_photo_path = os.path.join(STATIC_FOLDER, suspect.photo_path)
            img = cv2.imread(full_photo_path)
            if img is None:
                raise ValueError(
                    f"cv2.imread returned None for path: {full_photo_path!r}"
                )

            fa = get_app()
            gallery: list[np.ndarray] = []

            # Step 4: Extract base embedding from the original photo
            base_emb, _ = get_face_embedding(fa, img)
            if base_emb is not None:
                gallery.append(base_emb)

            # Step 5a: Brightness/contrast augmentations (4 variants)
            # alpha controls contrast, beta controls brightness
            bc_params = [
                (0.8, -20),
                (0.9, -10),
                (1.1,  10),
                (1.2,  20),
            ]
            for alpha, beta in bc_params:
                augmented = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                emb, _ = get_face_embedding(fa, augmented)
                if emb is not None:
                    gallery.append(emb)

            # Step 5b: Gaussian blur augmentations (2 variants)
            blur_kernels = [(3, 3), (5, 5)]
            for kernel in blur_kernels:
                blurred = cv2.GaussianBlur(img, kernel, 0)
                emb, _ = get_face_embedding(fa, blurred)
                if emb is not None:
                    gallery.append(emb)

            # Step 8/9: Persist results
            if not gallery:
                suspect.enroll_status = "failed"
                suspect.error_msg = "No face detected"
                logger.error(
                    "enroll_single: No face detected for suspect %d (%r).",
                    suspect_id,
                    suspect.name,
                )
            else:
                suspect.gallery = pickle.dumps(gallery)
                suspect.enroll_status = "ready"
                logger.info(
                    "enroll_single: Enrolled suspect %d (%r) with %d embeddings.",
                    suspect_id,
                    suspect.name,
                    len(gallery),
                )

            db.session.commit()

        except Exception as exc:
            logger.error(
                "enroll_single: Exception while enrolling suspect %d: %s\n%s",
                suspect_id,
                exc,
                traceback.format_exc(),
            )
            try:
                suspect.enroll_status = "failed"
                suspect.error_msg = str(exc)
                db.session.commit()
            except Exception:
                logger.error(
                    "enroll_single: Failed to persist error state for suspect %d.\n%s",
                    suspect_id,
                    traceback.format_exc(),
                )


def enroll_group_image(app, temp_path: str) -> None:
    """Background thread target: detect all faces in group photo; create one Suspect per face.

    1. Push an app context
    2. Read the group photo from temp_path
    3. Detect all faces using fa.get(img) (not get_face_embedding — we want ALL faces)
    4. If no faces detected: log warning, delete temp file, return
    5. For each detected face:
       a. Crop the face region from the image using face.bbox [x1, y1, x2, y2]
       b. Save the crop to UPLOAD_CROPS directory with a UUID filename
       c. L2-normalise the face embedding
       d. Create a Suspect record with:
          - name = f"Unknown_{uuid4().hex[:8]}"
          - region = "Unknown"
          - photo_path = _to_relative(crop_save_path)
          - gallery = pickle.dumps([normalised_embedding])
          - enroll_mode = 'structured'
          - enroll_status = 'ready'
       e. Add to session
    6. Commit all new Suspect records
    7. Delete the temporary file (os.remove(temp_path))
    8. On any exception: log ERROR with traceback, try to delete temp file
    """
    # Import inside function body to avoid circular imports at module level.
    from crimai.models import Suspect, db

    with app.app_context():
        try:
            img = cv2.imread(temp_path)
            if img is None:
                raise ValueError(
                    f"cv2.imread returned None for path: {temp_path!r}"
                )

            fa = get_app()
            faces = fa.get(img)

            if not faces:
                logger.warning(
                    "enroll_group_image: No faces detected in group photo %r; skipping DB writes.",
                    temp_path,
                )
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
                return

            img_h, img_w = img.shape[:2]

            for face in faces:
                # Crop the face region, clamping to image bounds
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_w, x2)
                y2 = min(img_h, y2)
                crop = img[y1:y2, x1:x2]

                # Save crop to UPLOAD_CROPS directory
                crop_filename = f"group_{uuid.uuid4().hex}.jpg"
                crop_path = os.path.join(UPLOAD_CROPS, crop_filename)
                cv2.imwrite(crop_path, crop)

                # L2-normalise the face embedding
                emb = face.embedding.astype(np.float32)
                norm = np.linalg.norm(emb)
                if norm == 0.0:
                    logger.warning(
                        "enroll_group_image: Zero-norm embedding for a face in %r; skipping this face.",
                        temp_path,
                    )
                    continue
                emb_normalised = emb / norm

                # Create a Suspect record
                suspect = Suspect(
                    name=f"Unknown_{uuid.uuid4().hex[:8]}",
                    region="Unknown",
                    photo_path=_to_relative(crop_path),
                    gallery=pickle.dumps([emb_normalised]),
                    enroll_mode="structured",
                    enroll_status="ready",
                )
                db.session.add(suspect)

            db.session.commit()
            logger.info(
                "enroll_group_image: Enrolled %d face(s) from group photo %r.",
                len(faces),
                temp_path,
            )

        except Exception as exc:
            logger.error(
                "enroll_group_image: Exception while processing group photo %r: %s\n%s",
                temp_path,
                exc,
                traceback.format_exc(),
            )
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass


def process_media(app, media_id: int) -> None:
    """Background thread target: run full detection pipeline on a Media record.

    Image branch:
    1. Push app context
    2. Load Media record; set status='running', commit
    3. Call load_all_embeddings() to get all ready suspect galleries
    4. Read the image from media.file_path (relative to static folder)
    5. Apply CLAHE preprocessing via preprocess_frame()
    6. Detect faces via fa.get(preprocessed_img)
    7. For each detected face:
       a. Crop the face region, save to UPLOAD_CROPS with UUID filename
       b. L2-normalise the embedding
       c. Call match_embedding(emb, suspect_embeddings)
       d. If match found: create DetectionResult
       e. If no match: create UnknownIdentity with pickled embedding
    8. Set media.status='done', media.progress=100, media.finished_at, media.processed=True
    9. Commit
    10. On unhandled exception: set media.status='failed', media.error_msg=str(e), log ERROR

    Video branch will be added in task 7.2 — for now, only handle file_type='image'.
    For file_type='video', set status='failed' with error_msg='Video processing not yet implemented'.
    """
    # Import inside function body to avoid circular imports at module level.
    from crimai.models import DetectionResult, Media, UnknownIdentity, db

    with app.app_context():
        media = db.session.get(Media, media_id)
        if media is None:
            logger.error(
                "process_media: Media %d not found in database.", media_id
            )
            return

        try:
            # Step 2: Mark as running
            media.status = "running"
            db.session.commit()

            # Step 3: Load all suspect embeddings
            suspect_embeddings = load_all_embeddings()

            # Handle video branch
            if media.file_type == "video":
                _process_video(app, media, db, suspect_embeddings)
                return

            # Step 4: Read the image
            img_path = os.path.join(STATIC_FOLDER, media.file_path)
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(
                    f"cv2.imread returned None for path: {img_path!r}"
                )

            # Step 5: Apply CLAHE preprocessing
            preprocessed_img = preprocess_frame(img)

            # Step 6: Detect faces
            fa = get_app()
            faces = fa.get(preprocessed_img)

            img_h, img_w = img.shape[:2]
            os.makedirs(UPLOAD_CROPS, exist_ok=True)

            # Step 7: Process each detected face
            for face in faces:
                # Step 7a: Crop the face region
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_w, x2)
                y2 = min(img_h, y2)
                crop = img[y1:y2, x1:x2]

                crop_filename = f"det_{uuid.uuid4().hex}.jpg"
                crop_path = os.path.join(UPLOAD_CROPS, crop_filename)
                cv2.imwrite(crop_path, crop)

                # Step 7b: L2-normalise the embedding
                emb = face.embedding.astype(np.float32)
                norm = np.linalg.norm(emb)
                if norm == 0.0:
                    logger.warning(
                        "process_media: Zero-norm embedding for a face in media %d; skipping.",
                        media_id,
                    )
                    continue
                emb = emb / norm

                # Step 7c: Match against suspect galleries
                label, score = match_embedding(emb, suspect_embeddings)

                if label is not None:
                    # Step 7d: Match found — create DetectionResult
                    suspect_id = int(label.split(":")[0])
                    detection = DetectionResult(
                        suspect_id=suspect_id,
                        media_id=media_id,
                        confidence=score,
                        frame_number=0,
                        timestamp_sec=0.0,
                        crop_path=_to_relative(crop_path),
                        frame_path=_to_relative(img_path),
                    )
                    db.session.add(detection)
                    logger.info(
                        "process_media: Match found for media %d — suspect %d (score=%.3f).",
                        media_id,
                        suspect_id,
                        score,
                    )
                else:
                    # Step 7e: No match — create UnknownIdentity
                    unknown = UnknownIdentity(
                        media_id=media_id,
                        crop_path=_to_relative(crop_path),
                        embedding=pickle.dumps(emb),
                        best_score=score,
                        closest_suspect=label if label is not None else "None",
                    )
                    db.session.add(unknown)
                    logger.info(
                        "process_media: No match for a face in media %d (best_score=%.3f).",
                        media_id,
                        score,
                    )

            # Step 8: Mark as done
            media.status = "done"
            media.progress = 100
            media.finished_at = datetime.utcnow()
            media.processed = True
            db.session.commit()

            logger.info(
                "process_media: Completed processing media %d (%d face(s) detected).",
                media_id,
                len(faces) if faces else 0,
            )

        except Exception as exc:
            logger.error(
                "process_media: Exception while processing media %d: %s\n%s",
                media_id,
                exc,
                traceback.format_exc(),
            )
            try:
                media.status = "failed"
                media.error_msg = str(exc)
                db.session.commit()
            except Exception:
                logger.error(
                    "process_media: Failed to persist error state for media %d.\n%s",
                    media_id,
                    traceback.format_exc(),
                )

def _draw_corner_accents(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    length: int = 15,
    thickness: int = 3,
) -> None:
    """Draw L-shaped corner accents at each corner of a bounding box.

    Four corners: top-left, top-right, bottom-left, bottom-right.
    Each corner gets a horizontal and a vertical line of `length` pixels.
    """
    # Top-left corner
    cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)
    # Top-right corner
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)
    # Bottom-left corner
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)
    # Bottom-right corner
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)


def _process_video(
    app,
    media,
    db,
    suspect_embeddings: dict[str, list[np.ndarray]],
) -> None:
    """Process a video Media record: sample frames, detect/match faces, annotate, re-encode.

    Called from within an active app context (pushed by process_media).

    Steps:
    1. Open the video with cv2.VideoCapture
    2. Get video properties: fps, width, height, total_frames
    3. Create a temp AVI output path in UPLOAD_OUTPUT
    4. Create cv2.VideoWriter with XVID codec
    5. Iterate frames:
       - Read frame; on failure break
       - Write non-sampled frames directly to output
       - For sampled frames (frame_idx % FRAME_SKIP == 0):
         * Apply CLAHE
         * Detect faces
         * Match embeddings; create DetectionResult / UnknownIdentity records
         * Draw bounding boxes and corner accents
         * Update alert_frames_remaining counter
       - Draw alert overlay when alert_frames_remaining > 0
       - Update media.progress every 50 frames
       - Write annotated frame to VideoWriter
    6. Release VideoCapture and VideoWriter
    7. Re-encode with ffmpeg to H.264 MP4
    8. On ffmpeg success: use mp4_path, delete avi_path
    9. On ffmpeg failure: log stderr, use avi_path, set warning in error_msg
    10. Set media.output_path, status='done', progress=100, finished_at, processed=True
    11. Commit

    Requirements: 6.6, 7.3, 7.7, 7.8, 7.9, 7.10, 7.11, 7.12, 7.13
    """
    # Import inside function body to avoid circular imports at module level.
    from crimai.models import DetectionResult, UnknownIdentity

    media_id = media.id
    video_path = os.path.join(STATIC_FOLDER, media.file_path)

    # Step 1: Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cv2.VideoCapture failed to open: {video_path!r}")

    # Step 2: Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Step 3: Create temp AVI output path
    avi_filename = f"out_{uuid.uuid4().hex}.avi"
    avi_path = os.path.join(UPLOAD_OUTPUT, avi_filename)
    os.makedirs(UPLOAD_OUTPUT, exist_ok=True)
    os.makedirs(UPLOAD_CROPS, exist_ok=True)

    # Step 4: Create VideoWriter with XVID codec
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(avi_path, fourcc, fps, (width, height))

    fa = get_app()
    alert_frames_remaining = 0
    frame_idx = 0

    try:
        # Step 5: Iterate frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # For sampled frames: apply CLAHE, detect, match, annotate
            if frame_idx % FRAME_SKIP == 0:
                preprocessed = preprocess_frame(frame)
                faces = fa.get(preprocessed)

                img_h, img_w = frame.shape[:2]
                match_found_this_frame = False

                for face in faces:
                    x1, y1, x2, y2 = [int(v) for v in face.bbox]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_w, x2)
                    y2 = min(img_h, y2)

                    # L2-normalise embedding
                    emb = face.embedding.astype(np.float32)
                    norm = np.linalg.norm(emb)
                    if norm == 0.0:
                        logger.warning(
                            "_process_video: Zero-norm embedding at frame %d in media %d; skipping face.",
                            frame_idx,
                            media_id,
                        )
                        continue
                    emb = emb / norm

                    label, score = match_embedding(emb, suspect_embeddings)

                    # Save face crop
                    crop = frame[y1:y2, x1:x2]
                    crop_filename = f"vid_{uuid.uuid4().hex}.jpg"
                    crop_path = os.path.join(UPLOAD_CROPS, crop_filename)
                    cv2.imwrite(crop_path, crop)

                    if label is not None:
                        # Match found — green box with label
                        suspect_id_val = int(label.split(":")[0])
                        suspect_name = label.split(":", 1)[1]
                        timestamp_sec = frame_idx / fps

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        _draw_corner_accents(frame, x1, y1, x2, y2, (0, 255, 0))
                        cv2.putText(
                            frame,
                            f"{suspect_name} ({score:.2f})",
                            (x1, max(y1 - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 0),
                            2,
                        )

                        detection = DetectionResult(
                            suspect_id=suspect_id_val,
                            media_id=media_id,
                            confidence=score,
                            frame_number=frame_idx,
                            timestamp_sec=timestamp_sec,
                            crop_path=_to_relative(crop_path),
                            frame_path=_to_relative(video_path),
                        )
                        db.session.add(detection)
                        match_found_this_frame = True

                        logger.info(
                            "_process_video: Match at frame %d — suspect %d (score=%.3f).",
                            frame_idx,
                            suspect_id_val,
                            score,
                        )
                    else:
                        # No match — grey box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
                        _draw_corner_accents(frame, x1, y1, x2, y2, (128, 128, 128))

                        unknown = UnknownIdentity(
                            media_id=media_id,
                            crop_path=_to_relative(crop_path),
                            embedding=pickle.dumps(emb),
                            best_score=score,
                            closest_suspect="None",
                        )
                        db.session.add(unknown)

                # Trigger alert banner for subsequent frames
                if match_found_this_frame:
                    alert_frames_remaining = ALERT_DURATION_FRAMES

            # Draw alert overlay banner when active
            if alert_frames_remaining > 0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                cv2.putText(
                    frame,
                    "SUSPECT DETECTED",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 255, 255),
                    2,
                )
                alert_frames_remaining -= 1

            # Update progress every 50 frames
            if frame_idx % 50 == 0 and total_frames > 0:
                media.progress = min(99, int(frame_idx / total_frames * 100))
                db.session.commit()

            # Write annotated frame
            writer.write(frame)
            frame_idx += 1

    finally:
        cap.release()
        writer.release()

    # Step 7: Re-encode with ffmpeg to H.264 MP4
    mp4_filename = avi_filename.replace(".avi", ".mp4")
    mp4_path = os.path.join(UPLOAD_OUTPUT, mp4_filename)

    ffmpeg_result = None
    try:
        ffmpeg_result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", avi_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                mp4_path,
            ],
            capture_output=True,
        )
    except FileNotFoundError:
        logger.warning(
            "_process_video: ffmpeg not found on PATH for media %d; falling back to AVI.",
            media_id,
        )

    if ffmpeg_result is not None and ffmpeg_result.returncode == 0:
        # Step 8: ffmpeg succeeded — use MP4, delete AVI
        output_path = mp4_path
        try:
            os.remove(avi_path)
        except Exception:
            pass
        logger.info(
            "_process_video: ffmpeg re-encoding succeeded for media %d → %r.",
            media_id,
            mp4_path,
        )
    else:
        # Step 9: ffmpeg failed or not available — fall back to AVI
        if ffmpeg_result is not None:
            stderr_text = ffmpeg_result.stderr.decode("utf-8", errors="replace")
            logger.error(
                "_process_video: ffmpeg re-encoding failed for media %d. stderr:\n%s",
                media_id,
                stderr_text,
            )
        output_path = avi_path
        media.error_msg = "ffmpeg re-encoding failed; using raw AVI output"

    # Step 10: Finalise media record
    media.output_path = _to_relative(output_path)
    media.status = "done"
    media.progress = 100
    media.finished_at = datetime.utcnow()
    media.processed = True

    # Step 11: Commit
    db.session.commit()

    logger.info(
        "_process_video: Completed video processing for media %d. Output: %r.",
        media_id,
        output_path,
    )
