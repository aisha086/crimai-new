import streamlit as st
import cv2
import numpy as np
import sqlite3
import pickle
import os
import datetime
import tempfile
from pathlib import Path
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CrimAI — Suspect Recognition",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Constants ────────────────────────────────────────────────────────────────
DB_PATH            = "/home/claude/crimai_app/crimai.db"
SUSPECT_FRAMES_DIR = "/home/claude/crimai_app/suspect_frames/"
UNKNOWN_FRAMES_DIR = "/home/claude/crimai_app/unknown_frames/"
SIMILARITY_THRESH  = 0.50
FRAME_SKIP         = 3
ALERT_DURATION_FRAMES = 45

os.makedirs(SUSPECT_FRAMES_DIR, exist_ok=True)
os.makedirs(UNKNOWN_FRAMES_DIR, exist_ok=True)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base */
body, .stApp { background-color: #0d0d0d; color: #e0e0e0; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #111827 0%, #0f172a 100%);
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] .stRadio label { color: #94a3b8; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover { color: #f8fafc; }

/* Headers */
h1 { color: #f1f5f9 !important; letter-spacing: -0.5px; }
h2, h3 { color: #cbd5e1 !important; }

/* Cards */
.card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}
.alert-card {
    background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
    border: 1px solid #dc2626;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{border-color:#dc2626} 50%{border-color:#f87171} }

.suspect-card {
    background: #1e293b;
    border: 1px solid #475569;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}
.suspect-card:hover { border-color: #60a5fa; }

.metric-box {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #3b82f6; }
.metric-label { font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }

.status-dot-green { display:inline-block; width:10px; height:10px; background:#22c55e; border-radius:50%; margin-right:6px; }
.status-dot-red   { display:inline-block; width:10px; height:10px; background:#ef4444; border-radius:50%; animation:pulse 1s infinite; margin-right:6px; }
.status-dot-gray  { display:inline-block; width:10px; height:10px; background:#64748b; border-radius:50%; margin-right:6px; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
    color: white; border: none; border-radius: 8px;
    font-weight: 600; letter-spacing: 0.3px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(37,99,235,0.4);
}

/* Inputs */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSlider > div { background: #0f172a; border-color: #334155; color: #e2e8f0; }

/* Progress bar */
.stProgress > div > div { background-color: #2563eb; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #0f172a; border-radius: 8px; gap: 4px; }
.stTabs [data-baseweb="tab"] { color: #94a3b8; border-radius: 6px; }
.stTabs [aria-selected="true"] { background: #1e3a5f !important; color: #60a5fa !important; }

/* Hide default streamlit footer */
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Database ─────────────────────────────────────────────────────────────────
@st.cache_resource
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS suspects (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            label        TEXT UNIQUE,
            image_blob   BLOB,
            gallery      BLOB,
            enroll_mode  TEXT DEFAULT 'single',
            image_count  INTEGER DEFAULT 1,
            added_at     TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS match_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT,
            frame_number INTEGER,
            suspect_id   TEXT,
            confidence   REAL,
            alert_msg    TEXT,
            frame_path   TEXT
        )
    """)
    conn.commit()
    return conn


def load_face_model():
    """Load InsightFace model (cached)."""
    try:
        from insightface.app import FaceAnalysis
        fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        fa.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.35)
        return fa
    except Exception as e:
        return None


@st.cache_resource(show_spinner=False)
def get_face_model():
    return load_face_model()


def preprocess_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)


def get_face_embedding(fa, img_bgr):
    """Return (embedding, face_obj) or (None, None)."""
    for img_v in [preprocess_frame(img_bgr), img_bgr]:
        for thresh in [0.35, 0.20, 0.10]:
            fa.det_model.det_thresh = thresh
            faces = fa.get(img_v)
            if faces:
                fa.det_model.det_thresh = 0.35
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                emb = face.embedding.astype(np.float32)
                emb = emb / (np.linalg.norm(emb) + 1e-6)
                return emb, face
    fa.det_model.det_thresh = 0.35
    return None, None


def enroll_suspect(conn, fa, label, img_bgr, img_bytes):
    """Enroll a suspect with augmented gallery."""
    emb, face = get_face_embedding(fa, img_bgr)
    if emb is None:
        return False, "No face detected in the image."

    gallery = [emb]

    # Light augmentations for a richer gallery
    for al, be in [(1.3, 20), (0.8, -15), (1.5, 35), (0.6, -30)]:
        aug = cv2.convertScaleAbs(img_bgr, alpha=al, beta=be)
        e, _ = get_face_embedding(fa, aug)
        if e is not None:
            gallery.append(e)

    for k in [3, 5]:
        blur = cv2.GaussianBlur(img_bgr, (k, k), 0)
        e, _ = get_face_embedding(fa, blur)
        if e is not None:
            gallery.append(e)

    gallery_blob = pickle.dumps(gallery)
    now = datetime.datetime.now().isoformat()

    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO suspects (label, image_blob, gallery, enroll_mode, image_count, added_at)
        VALUES (?, ?, ?, 'single', ?, ?)
    """, (label, img_bytes, gallery_blob, 1, now))
    conn.commit()
    return True, f"✅ Enrolled **{label}** with {len(gallery)} gallery embeddings."


def get_all_suspects(conn):
    cur = conn.cursor()
    cur.execute("SELECT id, label, image_blob, image_count, added_at FROM suspects ORDER BY added_at DESC")
    return cur.fetchall()


def delete_suspect(conn, label):
    cur = conn.cursor()
    cur.execute("DELETE FROM suspects WHERE label=?", (label,))
    conn.commit()


def load_suspect_embeddings(conn):
    cur = conn.cursor()
    cur.execute("SELECT label, gallery FROM suspects")
    rows = cur.fetchall()
    embeddings = {}
    for label, gallery_blob in rows:
        if gallery_blob:
            embeddings[label] = pickle.loads(gallery_blob)
    return embeddings


def match_embedding(emb, suspect_embeddings):
    """Returns (best_label, best_score) or (None, 0)."""
    if not suspect_embeddings:
        return None, 0.0
    best_label, best_score = None, 0.0
    for label, gallery in suspect_embeddings.items():
        for ref in gallery:
            score = float(cosine_similarity(emb.reshape(1, -1), ref.reshape(1, -1))[0][0])
            if score > best_score:
                best_score = score
                best_label = label
    if best_score >= SIMILARITY_THRESH:
        return best_label, best_score
    return None, best_score


def draw_bbox(frame, face, label, score, color=(0, 80, 220)):
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    # Main rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Corner accents
    clen = 18
    thick = 3
    for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (cx, cy), (cx+dx*clen, cy), color, thick)
        cv2.line(frame, (cx, cy), (cx, cy+dy*clen), color, thick)

    # Label background
    tag = f" {label}  {score:.0%} "
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.55
    (tw, th), _ = cv2.getTextSize(tag, font, fs, 1)
    ty = y1 - 10 if y1 - 10 > th + 2 else y2 + th + 10
    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw, ty + 4), color, -1)
    cv2.putText(frame, tag, (x1, ty), font, fs, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_unknown_bbox(frame, face):
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)
    cv2.putText(frame, "Unknown", (x1, y1-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    return frame


def draw_alert_overlay(frame, detected_suspects):
    """Red flash overlay with suspect names."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 180), -1)
    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
    names = ", ".join(detected_suspects)
    cv2.putText(frame, f"⚠ SUSPECT DETECTED: {names}",
                (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 80, 80), 2, cv2.LINE_AA)
    return frame


def process_video(video_path, suspect_embeddings, conn, progress_cb=None, frame_cb=None):
    fa = get_face_model()
    if fa is None:
        return None, [], 0

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = video_path.replace(".mp4", "_annotated.mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    alerts = []
    alert_active_until = {}  # suspect_label -> frame number
    frame_num = 0
    cur = conn.cursor()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = frame.copy()

        if frame_num % FRAME_SKIP == 0:
            enhanced = preprocess_frame(frame)
            fa.det_model.det_thresh = 0.35
            faces = fa.get(enhanced)
            if not faces:
                faces = fa.get(frame)

            detected_suspects = []

            if faces:
                for face in faces:
                    emb = face.embedding.astype(np.float32)
                    emb = emb / (np.linalg.norm(emb) + 1e-6)
                    label, score = match_embedding(emb, suspect_embeddings)

                    if label:
                        detected_suspects.append(label)
                        alert_active_until[label] = frame_num + ALERT_DURATION_FRAMES
                        color = (0, 60, 220)
                        annotated = draw_bbox(annotated, face, label, score, color)

                        # Save evidence frame
                        frame_path = f"{SUSPECT_FRAMES_DIR}{label}_{frame_num}.jpg"
                        cv2.imwrite(frame_path, annotated)

                        ts = datetime.datetime.now().isoformat()
                        cur.execute("""
                            INSERT INTO match_log (timestamp, frame_number, suspect_id, confidence, alert_msg, frame_path)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (ts, frame_num, label, float(score),
                              f"Suspect {label} detected at frame {frame_num}", frame_path))
                        conn.commit()
                        alerts.append({
                            "frame": frame_num, "label": label, "score": score,
                            "time": f"{frame_num/fps:.1f}s", "path": frame_path
                        })
                    else:
                        annotated = draw_unknown_bbox(annotated, face)

            # Check active alerts (persist box for ALERT_DURATION_FRAMES)
            still_active = {l for l, until in alert_active_until.items() if frame_num <= until}
            if still_active:
                annotated = draw_alert_overlay(annotated, list(still_active))

        out.write(annotated)
        frame_num += 1

        if progress_cb and frame_num % 10 == 0:
            progress_cb(frame_num / max(total_frames, 1))
        if frame_cb and frame_num % 30 == 0:
            frame_cb(annotated)

    cap.release()
    out.release()
    return out_path, alerts, total_frames


# ─── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:10px 0 20px">
        <div style="font-size:2.4rem;">🎯</div>
        <div style="font-size:1.2rem; font-weight:700; color:#f1f5f9;">CrimAI</div>
        <div style="font-size:0.72rem; color:#64748b; letter-spacing:2px;">SUSPECT RECOGNITION</div>
    </div>
    """, unsafe_allow_html=True)

    nav = st.radio(
        "Navigation",
        ["📊 Dashboard", "👤 Manage Suspects", "🚨 Alert Log"],
        label_visibility="collapsed"
    )

    # Model status
    st.markdown("---")
    fa = get_face_model()
    if fa:
        st.markdown('<span class="status-dot-green"></span> **Face Engine Ready**', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-dot-red"></span> **Face Engine Offline**', unsafe_allow_html=True)
        st.caption("InsightFace not available. Enroll/match disabled.")

    # Quick stats
    conn = get_db()
    suspects = get_all_suspects(conn)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM match_log")
    alert_count = cur.fetchone()[0]

    st.markdown("---")
    col1, col2 = st.columns(2)
    col1.metric("Suspects", len(suspects))
    col2.metric("Alerts", alert_count)


# ─── Pages ────────────────────────────────────────────────────────────────────
conn = get_db()
fa = get_face_model()

# ══════════════════════════════════════════════════════
#  PAGE 1 — DASHBOARD (Video Analysis)
# ══════════════════════════════════════════════════════
if nav == "📊 Dashboard":
    st.markdown("## 📊 Video Analysis Dashboard")

    suspects = get_all_suspects(conn)
    if not suspects:
        st.warning("⚠️ No suspects enrolled yet. Go to **Manage Suspects** to add some first.")
    else:
        st.markdown(f'<div class="card">🗂️ <b>{len(suspects)}</b> suspect(s) loaded — ready for matching</div>',
                    unsafe_allow_html=True)

    st.markdown("### Upload CCTV / Video File")
    uploaded = st.file_uploader("Drop video here", type=["mp4", "avi", "mov", "mkv"],
                                label_visibility="collapsed")

    col_thresh, col_skip, _ = st.columns([2, 2, 3])
    with col_thresh:
        SIMILARITY_THRESH = st.slider("Match Threshold", 0.30, 0.80, 0.50, 0.05,
                                      help="Higher = stricter matching")
    with col_skip:
        FRAME_SKIP = st.slider("Frame Skip", 1, 8, 3,
                               help="Process every Nth frame (higher = faster)")

    if uploaded and suspects and fa:
        if st.button("🎬 Analyze Video", use_container_width=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            suspect_embeddings = load_suspect_embeddings(conn)

            st.markdown("---")
            st.markdown("### ⏳ Processing...")
            progress_bar = st.progress(0.0)
            live_frame   = st.empty()
            status_text  = st.empty()

            def update_progress(p):
                progress_bar.progress(min(p, 1.0))
                status_text.markdown(f"<small style='color:#64748b'>Processed {p*100:.0f}%</small>",
                                     unsafe_allow_html=True)

            def update_frame(bgr_frame):
                rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                live_frame.image(rgb, caption="Live Preview", use_container_width=True)

            out_path, alerts, total_frames = process_video(
                tmp_path, suspect_embeddings, conn,
                progress_cb=update_progress,
                frame_cb=update_frame
            )

            progress_bar.progress(1.0)
            status_text.markdown("✅ **Processing complete!**")

            # Summary metrics
            st.markdown("### 📈 Results")
            mc1, mc2, mc3 = st.columns(3)
            mc1.markdown(f"""<div class="metric-box">
                <div class="metric-value">{total_frames}</div>
                <div class="metric-label">Frames Scanned</div>
            </div>""", unsafe_allow_html=True)
            mc2.markdown(f"""<div class="metric-box">
                <div class="metric-value" style="color:#ef4444">{len(alerts)}</div>
                <div class="metric-label">Alerts Fired</div>
            </div>""", unsafe_allow_html=True)
            unique_ids = len(set(a["label"] for a in alerts))
            mc3.markdown(f"""<div class="metric-box">
                <div class="metric-value" style="color:#f59e0b">{unique_ids}</div>
                <div class="metric-label">Unique Suspects</div>
            </div>""", unsafe_allow_html=True)

            # Alert cards
            if alerts:
                st.markdown("### 🚨 Detection Alerts")
                for a in alerts[:20]:
                    st.markdown(f"""
                    <div class="alert-card">
                        🚨 <b>{a['label']}</b> detected at frame <b>{a['frame']}</b>
                        &nbsp;|&nbsp; Time: <b>{a['time']}</b>
                        &nbsp;|&nbsp; Confidence: <b>{a['score']:.0%}</b>
                    </div>""", unsafe_allow_html=True)

                    if os.path.exists(a["path"]):
                        evidence = cv2.imread(a["path"])
                        if evidence is not None:
                            evidence_rgb = cv2.cvtColor(evidence, cv2.COLOR_BGR2RGB)
                            st.image(evidence_rgb, caption=f"Evidence — {a['label']} @ {a['time']}",
                                     use_container_width=True)

            # Download annotated video
            if out_path and os.path.exists(out_path):
                with open(out_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download Annotated Video",
                        data=f,
                        file_name="crimai_annotated.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
            os.unlink(tmp_path)

    elif not fa:
        st.info("Face engine not available. Install `insightface` and `onnxruntime` to enable analysis.")


# ══════════════════════════════════════════════════════
#  PAGE 2 — MANAGE SUSPECTS
# ══════════════════════════════════════════════════════
elif nav == "👤 Manage Suspects":
    st.markdown("## 👤 Suspect Database")

    tab_add, tab_list = st.tabs(["➕ Enroll New Suspect", "📋 View All Suspects"])

    # ── Add Suspect ──────────────────────────────────────────────────────────
    with tab_add:
        st.markdown("### Enroll a Suspect")
        st.caption("Upload a clear frontal face photo. The system generates augmented gallery embeddings automatically.")

        col_form, col_preview = st.columns([1, 1])

        with col_form:
            name = st.text_input("Suspect Name / ID", placeholder="e.g. John Doe or SUSPECT_001")
            photo = st.file_uploader("Face Photo", type=["jpg", "jpeg", "png"],
                                     label_visibility="collapsed")

            if photo and name.strip():
                if st.button("🔐 Enroll Suspect", use_container_width=True):
                    if not fa:
                        st.error("Face engine not available.")
                    else:
                        img_bytes = photo.read()
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        with st.spinner("Extracting embeddings..."):
                            ok, msg = enroll_suspect(conn, fa, name.strip(), img_bgr, img_bytes)

                        if ok:
                            st.success(msg)
                            st.balloons()
                        else:
                            st.error(msg)
            elif photo and not name.strip():
                st.warning("Please enter a suspect name.")

        with col_preview:
            if photo:
                photo.seek(0)
                img = Image.open(photo)
                st.image(img, caption="Uploaded Photo", use_container_width=True)
            else:
                st.markdown("""
                <div style="border:2px dashed #334155; border-radius:10px; height:220px;
                            display:flex; align-items:center; justify-content:center; color:#475569;">
                    Photo preview
                </div>""", unsafe_allow_html=True)

    # ── View Suspects ─────────────────────────────────────────────────────────
    with tab_list:
        suspects = get_all_suspects(conn)
        if not suspects:
            st.info("No suspects enrolled yet.")
        else:
            st.markdown(f"**{len(suspects)} suspect(s) in database**")
            for sid, label, img_blob, img_count, added_at in suspects:
                with st.container():
                    c1, c2, c3 = st.columns([1, 3, 1])
                    with c1:
                        if img_blob:
                            nparr = np.frombuffer(img_blob, np.uint8)
                            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if img_bgr is not None:
                                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                                img_pil = Image.fromarray(img_rgb)
                                img_pil = img_pil.resize((80, 80))
                                st.image(img_pil)
                    with c2:
                        st.markdown(f"**{label}**")
                        st.caption(f"ID: {sid} &nbsp;|&nbsp; Added: {added_at[:10] if added_at else 'N/A'}")
                    with c3:
                        if st.button("🗑️ Delete", key=f"del_{sid}", use_container_width=True):
                            delete_suspect(conn, label)
                            st.rerun()
                    st.markdown('<hr style="border-color:#1e293b; margin:8px 0">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  PAGE 3 — ALERT LOG
# ══════════════════════════════════════════════════════
elif nav == "🚨 Alert Log":
    st.markdown("## 🚨 Alert Log")

    cur = conn.cursor()
    cur.execute("SELECT timestamp, frame_number, suspect_id, confidence, alert_msg, frame_path FROM match_log ORDER BY id DESC LIMIT 200")
    rows = cur.fetchall()

    if not rows:
        st.info("No alerts recorded yet. Run a video analysis first.")
    else:
        # Summary
        total = len(rows)
        unique = len(set(r[2] for r in rows))
        last_ts = rows[0][0][:19] if rows[0][0] else "N/A"

        mc1, mc2, mc3 = st.columns(3)
        mc1.markdown(f'<div class="metric-box"><div class="metric-value">{total}</div><div class="metric-label">Total Alerts</div></div>', unsafe_allow_html=True)
        mc2.markdown(f'<div class="metric-box"><div class="metric-value" style="color:#f59e0b">{unique}</div><div class="metric-label">Unique Suspects</div></div>', unsafe_allow_html=True)
        mc3.markdown(f'<div class="metric-box"><div class="metric-value" style="color:#22c55e; font-size:1rem">{last_ts}</div><div class="metric-label">Latest Alert</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        # Filter
        filter_label = st.text_input("Filter by suspect name", "")
        if st.button("🗑️ Clear All Alerts", type="secondary"):
            cur.execute("DELETE FROM match_log")
            conn.commit()
            st.rerun()

        filtered = [r for r in rows if filter_label.lower() in r[2].lower()] if filter_label else rows

        for ts, frame_num, suspect_id, confidence, alert_msg, frame_path in filtered:
            conf_pct = f"{confidence:.0%}" if confidence else "N/A"
            time_str = ts[:19] if ts else "N/A"
            st.markdown(f"""
            <div class="suspect-card">
                <span class="status-dot-red"></span>
                <b>{suspect_id}</b>
                &nbsp;&nbsp;
                <span style="color:#64748b">Frame {frame_num}</span>
                &nbsp;|&nbsp;
                <span style="color:#22c55e">Confidence: {conf_pct}</span>
                &nbsp;|&nbsp;
                <span style="color:#64748b">{time_str}</span>
            </div>""", unsafe_allow_html=True)

            if frame_path and os.path.exists(frame_path):
                with st.expander(f"View evidence — {suspect_id} @ frame {frame_num}"):
                    ev = cv2.imread(frame_path)
                    if ev is not None:
                        ev_rgb = cv2.cvtColor(ev, cv2.COLOR_BGR2RGB)
                        st.image(ev_rgb, use_container_width=True)
