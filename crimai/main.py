"""
CrimAI main blueprint.

All application routes (dashboard, suspects, detect, history, results,
reports, regions, reported cases) are defined here.

Requirements: 1.6, 3.1–3.8, 4.1–4.6, 5.1–5.5, 6.1–6.8, 7.1–7.13,
              8.1–8.8, 9.1–9.3, 10.1–10.4, 11.1–11.5, 12.1–12.5,
              13.1–13.5, 14.1–14.3
"""

from __future__ import annotations

import os
import threading
import uuid
from datetime import datetime, timedelta

from flask import (
    Blueprint,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required
from sqlalchemy import func

from crimai.config import (
    ALLOWED_IMAGE_EXT,
    ALLOWED_VIDEO_EXT,
    REGIONS,
    STATIC_FOLDER,
    UPLOAD_EVIDENCE,
    UPLOAD_SUSPECTS,
)
from crimai.face_engine import enroll_group_image, enroll_single, get_app, process_media
from crimai.models import CaseReport, DetectionResult, Media, Suspect, UnknownIdentity, db

main = Blueprint("main", __name__)


# ---------------------------------------------------------------------------
# Root redirect
# ---------------------------------------------------------------------------

@main.route("/")
@login_required
def index():
    """Redirect root to dashboard."""
    return redirect(url_for("main.dashboard"))


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@main.route("/dashboard")
@login_required
def dashboard():
    """KPI cards, recent detections, and chart data."""
    total_suspects = Suspect.query.count()
    total_media = Media.query.count()
    processed_media = Media.query.filter_by(processed=True).count()
    pending_media = Media.query.filter_by(status="pending").count()

    recent_detections = (
        DetectionResult.query.order_by(DetectionResult.detected_at.desc())
        .limit(5)
        .all()
    )

    # Region breakdown: list of [region, count] — plain lists for JSON serialisation
    region_counts = [
        [row[0] or "Unknown", row[1]]
        for row in db.session.query(Media.region, func.count())
        .group_by(Media.region)
        .all()
    ]

    # Monthly detection counts for the last 6 months — plain lists
    monthly_counts = [
        [label, count]
        for label, count in _get_monthly_counts(months=6)
    ]

    return render_template(
        "main/dashboard.html",
        total_suspects=total_suspects,
        total_media=total_media,
        processed_media=processed_media,
        pending_media=pending_media,
        recent_detections=recent_detections,
        region_counts=region_counts,
        monthly_counts=monthly_counts,
        now=datetime.utcnow(),
    )


def _get_monthly_counts(months: int = 6) -> list[tuple[str, int]]:
    """Return list of (month_str, count) for the last *months* months.

    Uses SQLite's ``strftime('%Y-%m', uploaded_at)`` to group Media records.
    """
    rows = (
        db.session.query(
            func.strftime("%Y-%m", Media.uploaded_at).label("month"),
            func.count().label("cnt"),
        )
        .group_by("month")
        .order_by("month")
        .all()
    )

    # Build a dict for quick lookup
    counts_by_month: dict[str, int] = {row.month: row.cnt for row in rows}

    # Generate the last *months* month labels
    result: list[tuple[str, int]] = []
    now = datetime.utcnow()
    for i in range(months - 1, -1, -1):
        # Go back i months from now
        target = now - timedelta(days=30 * i)
        label = target.strftime("%Y-%m")
        result.append((label, counts_by_month.get(label, 0)))

    return result


# ---------------------------------------------------------------------------
# Suspects
# ---------------------------------------------------------------------------

@main.route("/suspects")
@login_required
def suspects():
    """Paginated suspect list."""
    page = request.args.get("page", 1, type=int)
    pagination = Suspect.query.order_by(Suspect.added_at.desc()).paginate(
        page=page, per_page=20, error_out=False
    )
    any_processing = Suspect.query.filter_by(enroll_status="processing").count() > 0

    return render_template(
        "main/suspects.html",
        pagination=pagination,
        suspects=pagination.items,
        any_processing=any_processing,
        regions=REGIONS,
    )


@main.route("/suspects/enroll", methods=["POST"])
@login_required
def enroll_suspect():
    """Single-photo suspect enrollment."""
    file = request.files.get("photo")
    if not file or not file.filename:
        flash("No file selected.", "error")
        return redirect(url_for("main.suspects"))

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_IMAGE_EXT:
        flash("Invalid file type. Allowed: jpg, jpeg, png.", "error")
        return redirect(url_for("main.suspects"))

    filename = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(UPLOAD_SUSPECTS, filename)
    file.save(save_path)

    suspect = Suspect(
        name=request.form.get("name", "Unknown"),
        region=request.form.get("region", "Unknown"),
        photo_path=f"uploads/suspects/{filename}",
        enroll_status="processing",
    )
    db.session.add(suspect)
    db.session.commit()

    app = current_app._get_current_object()
    t = threading.Thread(target=enroll_single, args=(app, suspect.id), daemon=True)
    t.start()

    flash("Enrollment started.", "success")
    return redirect(url_for("main.suspects"))


@main.route("/suspects/enroll-group", methods=["POST"])
@login_required
def enroll_group():
    """Group-photo enrollment — detect all faces and create one Suspect per face."""
    file = request.files.get("photo")
    if not file or not file.filename:
        flash("No file selected.", "error")
        return redirect(url_for("main.suspects"))

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_IMAGE_EXT:
        flash("Invalid file type. Allowed: jpg, jpeg, png.", "error")
        return redirect(url_for("main.suspects"))

    # Save to a temporary path; the background thread will delete it when done
    temp_filename = f"group_tmp_{uuid.uuid4().hex}.{ext}"
    temp_path = os.path.join(UPLOAD_SUSPECTS, temp_filename)
    file.save(temp_path)

    app = current_app._get_current_object()
    t = threading.Thread(
        target=enroll_group_image, args=(app, temp_path), daemon=True
    )
    t.start()

    flash("Group enrollment started. Suspects will appear shortly.", "success")
    return redirect(url_for("main.suspects"))


@main.route("/suspects/delete/<int:suspect_id>", methods=["POST"])
@login_required
def delete_suspect(suspect_id: int):
    """Delete a suspect and their photo from disk."""
    suspect = Suspect.query.get_or_404(suspect_id)

    # Remove photo file from disk (best-effort)
    if suspect.photo_path:
        full_path = os.path.join(STATIC_FOLDER, suspect.photo_path)
        try:
            if os.path.isfile(full_path):
                os.remove(full_path)
        except OSError:
            pass  # Non-fatal; DB record is still deleted

    db.session.delete(suspect)
    db.session.commit()

    flash("Suspect deleted.", "success")
    return redirect(url_for("main.suspects"))


# ---------------------------------------------------------------------------
# Detect / upload evidence
# ---------------------------------------------------------------------------

@main.route("/detect", methods=["GET"])
@login_required
def detect():
    """Render the evidence upload form."""
    return render_template("main/detect.html", regions=REGIONS)


@main.route("/detect", methods=["POST"])
@login_required
def detect_post():
    """Process uploaded evidence files."""
    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        flash("No files selected.", "error")
        return redirect(url_for("main.detect"))

    # Shared metadata from the form
    case_id = request.form.get("case_id", "")
    priority = request.form.get("priority", "Medium")
    description = request.form.get("description", "")
    location = request.form.get("location", "")
    region = request.form.get("region", "Unknown")
    inspector = request.form.get("inspector", "")
    upload_mode = request.form.get("upload_mode", "open_search")

    app = current_app._get_current_object()
    accepted = 0

    for file in files:
        if not file or not file.filename:
            continue

        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""

        if ext in ALLOWED_IMAGE_EXT:
            file_type = "image"
        elif ext in ALLOWED_VIDEO_EXT:
            file_type = "video"
        else:
            flash(f"Skipped '{file.filename}': unsupported file type.", "warning")
            continue

        filename = f"{uuid.uuid4().hex}.{ext}"
        save_path = os.path.join(UPLOAD_EVIDENCE, filename)
        file.save(save_path)

        media = Media(
            filename=file.filename,
            file_path=f"uploads/evidence/{filename}",
            file_type=file_type,
            case_id=case_id,
            priority=priority,
            description=description,
            location=location,
            region=region,
            inspector=inspector,
            upload_mode=upload_mode,
            status="pending",
        )
        db.session.add(media)
        db.session.commit()

        # Dispatch processing
        if file_type == "video":
            t = threading.Thread(
                target=process_media, args=(app, media.id), daemon=True
            )
            t.start()
        else:
            # Images are processed in a background thread too so the request
            # returns quickly, but they could also be processed synchronously.
            t = threading.Thread(
                target=process_media, args=(app, media.id), daemon=True
            )
            t.start()

        accepted += 1

    if accepted:
        flash(f"{accepted} file(s) uploaded and queued for processing.", "success")
    return redirect(url_for("main.history"))


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@main.route("/history")
@login_required
def history():
    """All media records, newest first."""
    media_records = Media.query.order_by(Media.uploaded_at.desc()).all()
    return render_template("main/history.html", media_records=media_records)


# ---------------------------------------------------------------------------
# Results viewer
# ---------------------------------------------------------------------------

@main.route("/results/<int:media_id>")
@login_required
def results(media_id: int):
    """Results viewer for a single media record."""
    media = Media.query.get_or_404(media_id)
    detections = (
        DetectionResult.query.filter_by(media_id=media_id)
        .order_by(DetectionResult.detected_at.asc())
        .all()
    )
    unknowns = (
        UnknownIdentity.query.filter_by(media_id=media_id)
        .order_by(UnknownIdentity.detected_at.asc())
        .all()
    )
    report = CaseReport.query.filter_by(media_id=media_id).first()

    return render_template(
        "main/results.html",
        media=media,
        detections=detections,
        unknowns=unknowns,
        report=report,
    )


# ---------------------------------------------------------------------------
# Job-status API
# ---------------------------------------------------------------------------

@main.route("/job-status/<int:media_id>")
@login_required
def job_status(media_id: int):
    """JSON status endpoint polled by the results page."""
    media = db.session.get(Media, media_id)
    if media is None:
        return jsonify({"error": "Not found"}), 404

    return jsonify(
        {
            "status": media.status,
            "progress": media.progress,
            "output_path": media.output_path,
            "error_msg": media.error_msg,
        }
    )


# ---------------------------------------------------------------------------
# Reports (aggregated stats)
# ---------------------------------------------------------------------------

@main.route("/reports")
@login_required
def reports():
    """Aggregated statistics dashboard."""
    total_detections = DetectionResult.query.count()

    avg_confidence_row = db.session.query(
        func.avg(DetectionResult.confidence)
    ).scalar()
    avg_confidence = round(float(avg_confidence_row or 0), 3)

    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)

    detections_this_week = DetectionResult.query.filter(
        DetectionResult.detected_at >= week_ago
    ).count()
    detections_this_month = DetectionResult.query.filter(
        DetectionResult.detected_at >= month_ago
    ).count()

    # Priority distribution: list of [priority, count] — plain lists for JSON
    priority_distribution = [
        [row[0] or "Unknown", row[1]]
        for row in db.session.query(Media.priority, func.count())
        .group_by(Media.priority)
        .all()
    ]

    # Monthly trend for last 6 months — plain lists
    monthly_trend = [
        [label, count]
        for label, count in _get_monthly_counts(months=6)
    ]

    # Region breakdown: list of [region, count] — plain lists
    region_breakdown = [
        [row[0] or "Unknown", row[1]]
        for row in db.session.query(Media.region, func.count())
        .group_by(Media.region)
        .all()
    ]

    # Last 20 detections
    last_detections = (
        DetectionResult.query.order_by(DetectionResult.detected_at.desc())
        .limit(20)
        .all()
    )

    return render_template(
        "main/reports.html",
        total_detections=total_detections,
        avg_confidence=avg_confidence,
        detections_this_week=detections_this_week,
        detections_this_month=detections_this_month,
        priority_distribution=priority_distribution,
        monthly_trend=monthly_trend,
        region_breakdown=region_breakdown,
        last_detections=last_detections,
    )


# ---------------------------------------------------------------------------
# Custom case report editor
# ---------------------------------------------------------------------------

@main.route("/report/custom/<int:media_id>")
@login_required
def report_editor(media_id: int):
    """Render the rich-text case report editor."""
    media = Media.query.get_or_404(media_id)
    report = CaseReport.query.filter_by(media_id=media_id).first()

    # Build a default HTML template pre-populated with case metadata
    if report is None:
        default_html = _build_default_report_html(media)
    else:
        default_html = report.html_content

    return render_template(
        "main/report_editor.html",
        media=media,
        report=report,
        default_html=default_html,
    )


@main.route("/report/custom/save/<int:media_id>", methods=["POST"])
@login_required
def save_report(media_id: int):
    """AJAX endpoint: create or update a CaseReport."""
    media = db.session.get(Media, media_id)
    if media is None:
        return jsonify({"success": False, "error": "Media not found"}), 404

    html_content = request.form.get("html_content", "")

    report = CaseReport.query.filter_by(media_id=media_id).first()
    if report is None:
        report = CaseReport(media_id=media_id, html_content=html_content)
        db.session.add(report)
    else:
        report.html_content = html_content
        report.updated_at = datetime.utcnow()

    db.session.commit()
    return jsonify({"success": True, "report_id": report.id})


def _build_default_report_html(media: Media) -> str:
    """Return a default HTML report template pre-filled with *media* metadata."""
    return (
        f"<h1>Case Report — {media.case_id or 'N/A'}</h1>"
        f"<p><strong>File:</strong> {media.filename}</p>"
        f"<p><strong>Inspector:</strong> {media.inspector or 'N/A'}</p>"
        f"<p><strong>Region:</strong> {media.region or 'N/A'}</p>"
        f"<p><strong>Priority:</strong> {media.priority or 'N/A'}</p>"
        f"<p><strong>Location:</strong> {media.location or 'N/A'}</p>"
        f"<p><strong>Description:</strong> {media.description or 'N/A'}</p>"
        f"<p><strong>Uploaded:</strong> {media.uploaded_at}</p>"
        "<h2>Findings</h2>"
        "<p>Enter your findings here.</p>"
    )


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------

@main.route("/regions")
@login_required
def regions():
    """Simple region list page."""
    return render_template("main/regions.html", regions=REGIONS)


# ---------------------------------------------------------------------------
# Reported cases
# ---------------------------------------------------------------------------

@main.route("/reported_cases")
@login_required
def reported_cases():
    """Media records that have an associated CaseReport, with optional filters."""
    status_filter = request.args.get("status", "")
    region_filter = request.args.get("region", "")

    query = Media.query.join(CaseReport, Media.id == CaseReport.media_id)

    if status_filter:
        query = query.filter(Media.status == status_filter)
    if region_filter:
        query = query.filter(Media.region == region_filter)

    media_records = query.order_by(Media.uploaded_at.desc()).all()

    return render_template(
        "main/reported_cases.html",
        media_records=media_records,
        regions=REGIONS,
        status_filter=status_filter,
        region_filter=region_filter,
    )
