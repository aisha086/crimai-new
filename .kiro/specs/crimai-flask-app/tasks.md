# Implementation Plan: CrimAI Flask Application

## Overview

Migrate and expand the existing Streamlit prototype into a production-grade Flask web application with multi-user authentication, SQLAlchemy ORM models, an asynchronous InsightFace processing pipeline, rich case-reporting, and a dark-themed responsive UI. Tasks are ordered so each step builds on the previous one, ending with full integration and tests.

## Tasks

- [x] 1. Project scaffolding and configuration
  - Create the `crimai/` package directory with an `__init__.py`
  - Create `crimai/config.py` with all constants: `MODEL_NAME`, `DET_SIZE_CPU`, `DET_SIZE_GPU`, `DET_THRESH`, `SIMILARITY_THRESH`, `FRAME_SKIP`, `ALERT_DURATION_FRAMES`, `STATIC_FOLDER`, `UPLOAD_EVIDENCE`, `UPLOAD_SUSPECTS`, `UPLOAD_CROPS`, `UPLOAD_OUTPUT`, `SECRET_KEY`, `ALLOWED_IMAGE_EXT`, `ALLOWED_VIDEO_EXT`, `REGIONS`
  - Create `crimai/static/uploads/evidence/`, `suspects/`, `crops/`, `output/` directories (add `.gitkeep` files)
  - Create `requirements.txt` pinning Flask, Flask-Login, Flask-SQLAlchemy, Werkzeug, insightface, onnxruntime, opencv-python-headless, numpy, hypothesis, pytest, pytest-flask
  - _Requirements: 2.6_

- [x] 2. SQLAlchemy data models
  - [x] 2.1 Create `crimai/models.py` with all six ORM models: `User`, `Suspect`, `Media`, `DetectionResult`, `UnknownIdentity`, `CaseReport`
    - Define all columns, defaults, and `db.relationship` entries with `cascade='all, delete-orphan'` as specified in the design
    - Enforce `unique=True` on `CaseReport.media_id` for the one-to-one constraint
    - _Requirements: 1.2, 1.8, 3.1, 3.6, 5.1, 6.2, 7.5, 7.6, 13.4, 16.1, 16.4, 17.1, 17.2, 17.3, 17.4_

  - [x] 2.2 Write unit tests for ORM cascade behaviour
    - Test that deleting a `Suspect` cascade-deletes its `DetectionResult` records
    - Test that deleting a `Media` cascade-deletes its `DetectionResult`, `UnknownIdentity`, and `CaseReport` records
    - Test the one-to-one `Media`↔`CaseReport` constraint
    - _Requirements: 17.1, 17.2, 17.3_

- [x] 3. Application factory
  - [x] 3.1 Create `crimai/app.py` with `create_app(config_object=None)` factory
    - Initialise SQLAlchemy (`db.init_app`), Flask-Login (`login_manager.init_app`), and call `db.create_all()` inside an app context
    - Call `os.makedirs` for all four upload directories with `exist_ok=True`
    - Pre-warm the FaceEngine by calling `get_app()` inside a try/except; log error and continue if it fails
    - Register the `auth` blueprint at `/auth` and the `main` blueprint at `/`
    - _Requirements: 2.1, 2.2, 2.4, 2.5, 18.1, 18.2, 18.3, 18.4_

- [x] 4. FaceEngine module
  - [x] 4.1 Create `crimai/face_engine.py` with the module-level `_fa` singleton and `get_app()` initialiser
    - Detect CUDA availability; use `CUDAExecutionProvider` + `det_size=(1024,1024)` if available, else `CPUExecutionProvider` + `det_size=(640,640)`
    - _Requirements: 2.2, 2.3, 2.4_

  - [x] 4.2 Implement `preprocess_frame()` and `get_face_embedding()`
    - `preprocess_frame`: apply CLAHE (clipLimit=3.0, tileGridSize=(8,8)) to the L channel of the LAB colour space
    - `get_face_embedding`: try preprocessed then raw frame; try det_thresh 0.35 → 0.20 → 0.10; return largest face embedding L2-normalised, or `(None, None)`
    - _Requirements: 7.2, 7.3_

  - [x] 4.3 Implement `load_all_embeddings()` and `match_embedding()`
    - `load_all_embeddings`: query all `Suspect` records with `enroll_status='ready'`; unpickle each gallery; return `{label: [embedding, ...]}`
    - `match_embedding`: iterate all galleries; compute cosine similarity; return `(best_label, best_score)` if score ≥ `SIMILARITY_THRESH`, else `(None, best_score)`
    - _Requirements: 7.1, 7.4_

  - [x] 4.4 Implement `_to_relative()` helper
    - Strip the `static/` prefix from an absolute path; return the remainder
    - Handle paths that are already relative (no-op)
    - _Requirements: 7.10, 18.5_

  - [x] 4.5 Write property test for `match_embedding()` — Property 4: Match Threshold Enforcement
    - **Property 4: Match Threshold Enforcement**
    - Generate random float32 embeddings and galleries with Hypothesis; assert label is non-None iff max cosine similarity ≥ `SIMILARITY_THRESH`
    - **Validates: Requirements 7.4**

  - [x] 4.6 Write property test for `match_embedding()` — Property 5: Cosine Similarity Bounds
    - **Property 5: Cosine Similarity Bounds**
    - Generate pairs of random L2-normalised float32 embeddings; assert returned score ∈ [-1.0, 1.0]
    - **Validates: Requirements 7.4**

  - [x] 4.7 Write property test for `_to_relative()` — Property 3: Relative Path Stripping
    - **Property 3: Relative Path Stripping**
    - Generate random filename strings; prepend the static prefix; assert `_to_relative()` strips it and that prepending the prefix reconstructs the original path
    - **Validates: Requirements 18.5, 18.6**

  - [x] 4.8 Write unit tests for `face_engine.py` helpers
    - Test `match_embedding()` returns `(None, 0.0)` when gallery is empty
    - Test `match_embedding()` returns the correct label when a single gallery entry exceeds the threshold
    - Test `_to_relative()` strips the static prefix, handles already-relative paths, and handles nested paths
    - _Requirements: 7.4, 18.5_

- [x] 5. Checkpoint — core modules complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Enrollment background thread targets
  - [x] 6.1 Implement `enroll_single(app, suspect_id)` in `face_engine.py`
    - Push an app context; load the `Suspect` record; call `get_face_embedding()` on the photo
    - Generate augmented variants (brightness/contrast × 4, Gaussian blur × 2); extract embeddings from each
    - Pickle the gallery list and write to `suspect.gallery`; set `enroll_status='ready'`
    - On any exception set `enroll_status='failed'` and `error_msg`; log at ERROR with traceback
    - _Requirements: 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

  - [x] 6.2 Implement `enroll_group_image(app, temp_path)` in `face_engine.py`
    - Push an app context; detect all faces in the group photo
    - For each face: crop the region, save to `uploads/crops/`, create a `Suspect` record with a generated name, pickle a single-embedding gallery, set `enroll_status='ready'`
    - Delete the temporary file; if no faces detected, log and return without DB writes
    - _Requirements: 4.2, 4.3, 4.4, 4.5, 4.6_

  - [x] 6.3 Write integration test for `enroll_single()`
    - After calling `enroll_single()` with a mock FaceEngine, assert `suspect.enroll_status == 'ready'` and `suspect.gallery` is non-None
    - _Requirements: 3.6_

- [x] 7. Media processing background thread target
  - [x] 7.1 Implement `process_media(app, media_id)` — image branch
    - Push app context; set `media.status='running'`; call `load_all_embeddings()`
    - Apply CLAHE; run face detection; for each face call `match_embedding()`
    - On match: save crop, create `DetectionResult`; on no match: save crop, create `UnknownIdentity` with pickled embedding
    - Set `media.status='done'`, `progress=100`, `finished_at`, `processed=True`
    - _Requirements: 6.5, 7.1, 7.2, 7.4, 7.5, 7.6, 7.11_

  - [x] 7.2 Implement `process_media(app, media_id)` — video branch
    - Sample frames at `FRAME_SKIP` interval; apply CLAHE; detect faces; match embeddings
    - Draw labelled bounding boxes with corner accents on matched faces; grey boxes on unknowns
    - Draw red alert overlay banner for `ALERT_DURATION_FRAMES` frames after each match
    - Update `media.progress` at regular intervals
    - Write annotated frames to an OpenCV `VideoWriter`; re-encode with ffmpeg to H.264 MP4
    - Store output path via `_to_relative()`; set `media.status='done'`, `progress=100`, `finished_at`, `processed=True`
    - On ffmpeg failure: log stderr, fall back to OpenCV output, set warning in `error_msg` but keep `status='done'`
    - On unhandled exception: set `media.status='failed'`, `media.error_msg=str(e)`; log ERROR with traceback
    - _Requirements: 6.6, 7.3, 7.7, 7.8, 7.9, 7.10, 7.11, 7.12, 7.13_

  - [x] 7.3 Write integration test for `process_media()`
    - After calling `process_media()` on a short synthetic test video with a mock FaceEngine, assert `media.status == 'done'` and at least one `DetectionResult` or `UnknownIdentity` record exists
    - _Requirements: 7.11_

- [x] 8. Auth blueprint
  - [x] 8.1 Create `crimai/auth.py` with the `auth` Flask blueprint
    - Implement `GET/POST /auth/register`: validate unique username, hash password with `generate_password_hash(method='pbkdf2:sha256')`, create `User`, flash success, redirect to login
    - Implement `GET/POST /auth/login`: look up user, call `check_password_hash`, call `login_user()`, redirect to dashboard; on failure flash error
    - Implement `GET /auth/logout`: call `logout_user()`, redirect to login
    - Implement `GET/POST /auth/forgot-password`: placeholder page with flash message
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 1.8_

  - [x] 8.2 Write unit tests for auth routes
    - Test register with unique username succeeds and creates a `User` record
    - Test register with duplicate username returns an error and does not create a duplicate
    - Test login with valid credentials creates a session and redirects to dashboard
    - Test login with invalid credentials does not create a session
    - Test logout invalidates the session and redirects to login
    - Test unauthenticated request to a protected route redirects to `/auth/login`
    - _Requirements: 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

  - [x] 8.3 Write property test for password hashing — Property 6: Password Hash Non-Reversibility
    - **Property 6: Password Hash Non-Reversibility**
    - Generate random password strings with Hypothesis; assert `password_hash != plaintext`, `check_password_hash(hash, password)` is True, and `check_password_hash(hash, other)` is False for any different string
    - **Validates: Requirements 1.2, 1.8**

- [x] 9. Checkpoint — auth and enrollment complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. Main blueprint — core routes
  - [x] 10.1 Create `crimai/main.py` with the `main` Flask blueprint; add `@login_required` to all routes
    - Implement `GET /` → redirect to `/dashboard`
    - Implement `GET /dashboard`: query KPI counts, last 5 `DetectionResult` records, region counts, monthly detection counts; pass to template
    - _Requirements: 1.6, 11.1, 11.2, 11.3, 11.4_

  - [x] 10.2 Implement suspect management routes
    - `GET /suspects`: paginated query of all `Suspect` records; pass to template; include auto-refresh meta tag when any suspect has `enroll_status='processing'`
    - `POST /suspects/enroll`: validate file extension; save photo to `uploads/suspects/`; create `Suspect(enroll_status='processing')`; dispatch `enroll_single` daemon thread; redirect to suspects page
    - `POST /suspects/enroll-group`: save to temp path; dispatch `enroll_group_image` daemon thread; redirect to suspects page
    - `POST /suspects/delete/<id>`: delete `Suspect` record (cascade handles DB children); delete photo file from disk; redirect to suspects page
    - _Requirements: 3.1, 3.2, 3.8, 4.1, 4.2, 5.1, 5.2, 5.3, 5.4_

  - [x] 10.3 Implement detect and history routes
    - `GET /detect`: render upload form
    - `POST /detect`: iterate uploaded files; validate extensions; save to `uploads/evidence/`; create `Media` records with metadata; dispatch `process_media` threads for video; process images synchronously; redirect to `/history`
    - `GET /history`: query all `Media` records ordered by `uploaded_at` desc; pass to template
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 10.1, 10.2, 10.3, 10.4_

  - [x] 10.4 Implement results, job-status, and report routes
    - `GET /results/<media_id>`: query `Media` with detections and unknowns; render results template
    - `GET /job-status/<media_id>`: return JSON `{status, progress, output_path, error_msg}`; 404 if not found
    - `GET /report/custom/<media_id>`: render report editor with default HTML template pre-populated with case metadata
    - `POST /report/custom/save/<media_id>`: create or update `CaseReport`; return JSON success; 404 if media not found
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 9.1, 9.2, 9.3, 13.1, 13.3, 13.4, 13.5_

  - [x] 10.5 Implement reports, regions, and reported cases routes
    - `GET /reports`: aggregate KPI totals, priority distribution, monthly trend, region breakdown table, last 20 detections; pass to template
    - `GET /regions`: pass `REGIONS` list to template
    - `GET /reported_cases`: query `Media` records that have an associated `CaseReport`; support filter by status and region
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 14.1, 14.2, 14.3_

  - [~] 10.6 Write unit tests for API routes
    - Test `GET /job-status/<id>` returns 200 JSON for an existing media record
    - Test `GET /job-status/<id>` returns 404 JSON for a missing media ID
    - Test `POST /report/custom/save/<id>` with valid HTML creates/updates a `CaseReport` record
    - Test `POST /report/custom/save/<id>` with a missing media ID returns 404
    - Test `POST /detect` assigns `file_type='image'` for image extensions and `file_type='video'` for video extensions
    - _Requirements: 6.2, 9.1, 9.2, 9.3, 13.4, 13.5_

- [x] 11. Checkpoint — all routes wired
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Base template and auth templates
  - [x] 12.1 Create `crimai/templates/base.html`
    - Dark theme via CSS custom properties; `<link>` to `static/css/style.css`; `<script>` for theme toggle (reads/writes `localStorage`; applies class before first paint to avoid flash)
    - Navigation bar with links to Dashboard, Suspects, Detect, History, Reports, Regions, Reported Cases, and Logout
    - Flash message block rendering success/error/info/warning categories
    - Responsive grid meta viewport tag
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5_

  - [x] 12.2 Create `crimai/templates/auth/login.html` and `auth/register.html`
    - Extend `base.html`; render login/register forms with CSRF-safe hidden fields; display flash errors inline
    - _Requirements: 1.1, 1.3, 1.5_

- [x] 13. Main page templates
  - [x] 13.1 Create `crimai/templates/main/dashboard.html`
    - KPI cards (total suspects, total media, processed, pending) with animated counter JS
    - Recent detections list (last 5)
    - Two `<canvas>` elements: bar chart (media by region) and line chart (detections by month)
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

  - [x] 13.2 Create `crimai/templates/main/suspects.html`
    - Paginated table with photo thumbnail, name, region, enroll-status badge (🟡/🟢/🔴), added date, delete button
    - Auto-refresh `<meta http-equiv="refresh" content="10">` when any suspect is processing
    - Enrollment form (single photo) and group-image enrollment form
    - Live search `<input>` that filters rows client-side
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 15.6_

  - [x] 13.3 Create `crimai/templates/main/detect.html`
    - Multi-file upload form with case metadata fields (case ID, priority, description, location, region, inspector)
    - Upload mode selector (open_search / dataset / group_image)
    - _Requirements: 6.1, 6.2, 6.3_

  - [x] 13.4 Create `crimai/templates/main/results.html`
    - Left sidebar: case metadata
    - Main column: HTML5 `<video>` player when done; progress spinner when running; error box when failed
    - DetectionResult grid cards (crop image, suspect name, timestamp, confidence bar)
    - Polling JS: `setInterval` every 3 s calling `/job-status/<id>`; on `status='done'` call `window.location.reload()`
    - Link to CaseReport editor
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8_

  - [x] 13.5 Create `crimai/templates/main/history.html`
    - Media cards: case ID, filename, inspector, region, priority badge, upload date, status badge, link to results
    - Live search input filtering by case ID, filename, inspector, region
    - Status filter control
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 15.6_

  - [x] 13.6 Create `crimai/templates/main/reports.html`
    - KPI totals row
    - Two `<canvas>` charts: priority donut and monthly trend line
    - Region breakdown table
    - Recent detections feed (last 20)
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

  - [x] 13.7 Create `crimai/templates/main/report_editor.html`
    - TinyMCE (CDN) rich text editor pre-populated with default HTML template
    - Save button triggers AJAX POST to `/report/custom/save/<media_id>`; shows success/error toast
    - _Requirements: 13.1, 13.2, 13.3_

  - [x] 13.8 Create `crimai/templates/main/regions.html` and `main/reported_cases.html`
    - `regions.html`: simple list of region names from `REGIONS`
    - `reported_cases.html`: Media cards with associated CaseReport; filter controls for status and region
    - _Requirements: 14.1, 14.2, 14.3_

- [x] 14. Static assets — CSS and JavaScript
  - [x] 14.1 Create `crimai/static/css/style.css`
    - Define CSS custom properties for dark theme colours; override set for light theme (`.light-mode` class)
    - Card, badge, button, form, table, progress-bar, flash-message, and nav styles
    - Responsive breakpoint: single-column layout below 1024 px
    - _Requirements: 15.1, 15.4_

  - [x] 14.2 Create `crimai/static/js/main.js`
    - Theme toggle: read `localStorage.getItem('theme')` on load; apply `.light-mode` to `<body>` before first paint; toggle on button click and persist
    - KPI counter animation: `requestAnimationFrame` loop counting from 0 to target value
    - Canvas chart helpers: `drawBarChart(ctx, labels, data, options)` and `drawLineChart(ctx, labels, data, options)` and `drawDonutChart(ctx, labels, data, options)` — no external libraries
    - Live search: `input` event listener on search fields; filter visible rows/cards by `textContent` match
    - _Requirements: 11.3, 11.4, 11.5, 12.2, 12.3, 15.2, 15.3, 15.6_

- [ ] 15. Test infrastructure and property-based tests
  - [~] 15.1 Create `tests/conftest.py`
    - Flask test client fixture using in-memory SQLite (`SQLALCHEMY_DATABASE_URI='sqlite://'`)
    - Temporary upload directories fixture using `tmp_path`
    - Mock FaceEngine fixture that returns a stub `FaceAnalysis` object
    - _Requirements: (test infrastructure)_

  - [~] 15.2 Write property tests for serialisation — Properties 1 and 2
    - **Property 1: Gallery Serialisation Round-Trip**
    - Generate lists of random float32 numpy arrays with Hypothesis; assert element-wise equality after `pickle.dumps` → `pickle.loads`
    - **Property 2: Embedding Serialisation Round-Trip**
    - Generate random float32 numpy arrays; assert element-wise equality after `pickle.dumps` → `pickle.loads`
    - Place both in `tests/test_properties.py` with `# Feature: crimai-flask-app, Property 1` and `Property 2` comments
    - **Validates: Requirements 16.1, 16.2, 16.3, 16.4, 16.5**

  - [~] 15.3 Consolidate all property tests into `tests/test_properties.py`
    - Move or import P3–P6 property tests (from tasks 4.5–4.7 and 8.3) into `tests/test_properties.py`
    - Annotate each with the `# Feature: crimai-flask-app, Property N` comment convention
    - Ensure minimum 100 iterations per property via `@settings(max_examples=100)`
    - _Requirements: 7.4, 16.3, 16.5, 18.5, 18.6, 1.2, 1.8_

- [~] 16. Final checkpoint — full test suite
  - Ensure all tests pass (`pytest tests/ -v`), ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Checkpoints at tasks 5, 9, 11, and 16 ensure incremental validation
- Property tests (P1–P6) use Hypothesis with `@settings(max_examples=100)` and validate universal correctness guarantees
- Unit tests validate specific examples and edge cases
- The FaceEngine is mocked in all unit tests to avoid loading the 300 MB ONNX model during CI
- All file paths stored in the database are relative to the static root; `_to_relative()` is the single point of conversion
