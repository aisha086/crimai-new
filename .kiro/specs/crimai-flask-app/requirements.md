# Requirements Document

## Introduction

CrimAI is a production-grade Flask web application for law enforcement surveillance and face detection. It migrates and expands an existing Streamlit prototype into a multi-user, role-authenticated system with a persistent database, asynchronous media processing pipeline, and rich reporting capabilities. The system allows investigators to enroll suspects into a watchlist, upload image and video evidence, run automated face recognition against the watchlist, and generate structured case reports — all through a dark-themed, responsive web interface.

---

## Glossary

- **System**: The CrimAI Flask web application as a whole.
- **Application**: The Flask WSGI application instance created by `create_app()`.
- **User**: An authenticated law enforcement operator interacting with the web interface.
- **Suspect**: A person enrolled in the watchlist database with one or more face embeddings.
- **Gallery**: A pickled list of 512-dimensional float32 numpy embeddings representing a single Suspect's face.
- **Media**: An uploaded evidence file (image or video) submitted for face recognition processing.
- **DetectionResult**: A database record representing a single confirmed match between a face in a Media file and a Suspect.
- **UnknownIdentity**: A database record representing an unmatched face extracted from a Media file.
- **CaseReport**: A rich-text HTML document associated with a single Media record.
- **FaceEngine**: The InsightFace `buffalo_l` ONNX model singleton used for face detection and embedding extraction.
- **Embedding**: A 512-dimensional float32 numpy vector representing a face, L2-normalised before storage and comparison.
- **Cosine Similarity**: The dot product of two L2-normalised embeddings, used as the match confidence score.
- **Similarity Threshold**: The minimum cosine similarity score required to classify a face as a known Suspect, defined in `config.py`.
- **Enrollment**: The process of extracting face embeddings from one or more photos and persisting them in a Suspect's Gallery.
- **Background Thread**: A Python daemon thread that runs AI-intensive work outside the HTTP request/response cycle.
- **App Context**: A Flask application context pushed inside a Background Thread to allow database access.
- **CLAHE**: Contrast Limited Adaptive Histogram Equalisation — the preprocessing step applied to frames before face detection.
- **Frame Sampling**: The process of selecting a subset of video frames for face detection based on a configurable skip interval.
- **Annotated Video**: A re-encoded video file with bounding boxes, labels, and alert overlays drawn on detected faces.
- **Priority**: A case urgency level assigned to a Media record: `Critical`, `High`, `Medium`, or `Low`.
- **Enroll Mode**: The method used to build a Suspect's Gallery: `single` (one photo) or `structured` (group image extraction).
- **Enroll Status**: The current state of a Suspect's Gallery build: `processing`, `ready`, or `failed`.
- **Job Status**: The current processing state of a Media record: `pending`, `running`, `done`, or `failed`.
- **Open Search**: A detection mode that matches uploaded Media against all enrolled Suspects.
- **Targeted Search**: A detection mode that matches uploaded Media against a single specified Suspect.
- **Region**: A geographic or administrative area tag attached to Suspects and Media records for filtering.
- **Inspector**: The law enforcement officer responsible for a Media submission.
- **Case ID**: A unique alphanumeric identifier assigned to a Media record by the submitting Inspector.
- **TinyMCE**: A browser-based rich text editor used for composing CaseReports.
- **ffmpeg**: A command-line multimedia tool invoked via `subprocess.run` to re-encode Annotated Videos to H.264/MP4.
- **KPI**: Key Performance Indicator — a summary statistic displayed on the Dashboard and Reports pages.
- **PBKDF2**: Password-Based Key Derivation Function 2 — the hashing algorithm used to store User passwords (`pbkdf2:sha256`).

---

## Requirements

### Requirement 1: User Authentication

**User Story:** As a law enforcement operator, I want to register and log in with a username and password, so that only authorised personnel can access the system.

#### Acceptance Criteria

1. THE System SHALL provide `/register`, `/login`, `/logout`, and `/forgot-password` HTTP routes.
2. WHEN a User submits a registration form with a unique username and password, THE System SHALL hash the password using `pbkdf2:sha256` and persist a new User record to the database.
3. IF a User submits a registration form with a username that already exists in the database, THEN THE System SHALL return a form error message without creating a duplicate record.
4. WHEN a User submits valid credentials on the login form, THE System SHALL create an authenticated session via Flask-Login and redirect the User to the Dashboard.
5. IF a User submits invalid credentials on the login form, THEN THE System SHALL return an error message and not create a session.
6. WHEN an unauthenticated request is made to any protected route, THE System SHALL redirect the request to `/login`.
7. WHEN a logged-in User requests `/logout`, THE System SHALL invalidate the session and redirect to `/login`.
8. THE System SHALL store only the password hash in the database and never store plaintext passwords.

---

### Requirement 2: Application Startup and FaceEngine Initialisation

**User Story:** As a system administrator, I want the face recognition model to be loaded once at startup, so that background processing threads always have immediate access to the model without per-request loading overhead.

#### Acceptance Criteria

1. WHEN the Application starts via `create_app()`, THE System SHALL initialise the FaceEngine singleton by calling `get_app()` exactly once.
2. THE FaceEngine SHALL load the InsightFace `buffalo_l` model using `CPUExecutionProvider` with `det_size=(640, 640)` and `det_thresh=0.35`.
3. WHERE a CUDA-capable GPU is available, THE FaceEngine SHALL use `CUDAExecutionProvider` with `det_size=(1024, 1024)`.
4. WHEN a Background Thread accesses the FaceEngine, THE System SHALL return the cached singleton without reloading the model.
5. IF the FaceEngine fails to initialise at startup, THEN THE System SHALL log the error and continue running with face recognition features disabled.
6. THE System SHALL define all tuneable constants — including Similarity Threshold, frame skip interval, and alert duration — in `config.py`, and all modules SHALL import constants exclusively from `config.py`.

---

### Requirement 3: Suspect Enrollment — Single Photo Mode

**User Story:** As an investigator, I want to enroll a suspect using a single frontal face photo, so that the system can recognise that person in future evidence uploads.

#### Acceptance Criteria

1. WHEN a User submits the enrollment form with a name, region, and photo file, THE System SHALL save the photo to disk and create a Suspect record with `enroll_status='processing'` before returning an HTTP response.
2. AFTER returning the HTTP response, THE System SHALL dispatch a Background Thread to build the Suspect's Gallery.
3. WHEN the Background Thread runs, THE System SHALL push an App Context and call `load_all_embeddings()` to access the database.
4. WHEN building the Gallery, THE System SHALL extract a base Embedding from the uploaded photo using the FaceEngine.
5. WHEN building the Gallery, THE System SHALL generate augmented variants of the photo (brightness/contrast adjustments and Gaussian blur) and extract an Embedding from each variant where a face is detected.
6. WHEN the Gallery build succeeds, THE System SHALL persist the Gallery as a pickled list of float32 numpy arrays in the `gallery` column and set `enroll_status='ready'`.
7. IF no face is detected in the uploaded photo, THEN THE System SHALL set `enroll_status='failed'` and store a descriptive error message.
8. THE System SHALL never perform AI inference on the HTTP request thread during enrollment.

---

### Requirement 4: Suspect Enrollment — Group Image Mode

**User Story:** As an investigator, I want to upload a group photo and have the system automatically create one suspect record per detected face, so that I can rapidly enroll multiple individuals from a single image.

#### Acceptance Criteria

1. WHEN a User submits a group image for enrollment, THE System SHALL save the photo to a temporary path and return an HTTP response before any face extraction begins.
2. AFTER returning the HTTP response, THE System SHALL dispatch a Background Thread that calls `enroll_group_image()`.
3. WHEN `enroll_group_image()` runs, THE System SHALL detect all faces in the group photo using the FaceEngine.
4. FOR EACH detected face, THE System SHALL crop the face region, create a new Suspect record with a system-generated name, save the cropped photo, and build a Gallery containing the face's Embedding.
5. WHEN all Suspect records have been created, THE System SHALL delete the temporary group photo file from disk.
6. IF no faces are detected in the group photo, THEN THE System SHALL log the failure and perform no database writes.

---

### Requirement 5: Suspect Watchlist Management

**User Story:** As an investigator, I want to view, search, and delete suspects from the watchlist, so that I can keep the database accurate and up to date.

#### Acceptance Criteria

1. THE System SHALL display all Suspect records in a paginated table showing photo thumbnail, name, region, Enroll Status badge, added date, and a delete button.
2. THE System SHALL render the Enroll Status badge as 🟡 Building when `enroll_status='processing'`, 🟢 Ready when `enroll_status='ready'`, and 🔴 Failed when `enroll_status='failed'`.
3. WHILE any Suspect record has `enroll_status='processing'`, THE System SHALL instruct the browser to auto-refresh the suspects page every 10 seconds.
4. WHEN a User requests deletion of a Suspect, THE System SHALL delete the Suspect record from the database, cascade-delete all associated DetectionResult records, and delete the Suspect's photo files from disk.
5. THE System SHALL provide a live search input that filters the displayed Suspect table by name or region without a full page reload.

---

### Requirement 6: Media Upload and Job Dispatch

**User Story:** As an investigator, I want to upload one or more image or video evidence files with case metadata, so that the system can process them for suspect matches.

#### Acceptance Criteria

1. WHEN a User submits the detect form, THE System SHALL accept multiple files in a single submission, each being an image (`jpg`, `jpeg`, `png`) or video (`mp4`, `avi`, `mov`, `mkv`).
2. FOR EACH uploaded file, THE System SHALL save the file to `uploads/evidence/`, create a Media record with the provided case metadata (priority, description, location, region, inspector name, case ID), and set `status='pending'`.
3. THE System SHALL support three upload modes: `open_search` (match against all Suspects), `dataset` (enroll and search specific Suspects), and `group_image` (extract unknowns then search).
4. WHEN the upload mode is `open_search` or `dataset`, THE System SHALL call `process_media()` for each file immediately after creating the Media record.
5. FOR image files, THE System SHALL process the image synchronously and set `processed=True` on the Media record before redirecting.
6. FOR video files, THE System SHALL dispatch processing to a Background Thread and redirect immediately; the Background Thread SHALL set `processed=True` upon completion.
7. AFTER all Media records are created and dispatched, THE System SHALL redirect the User to the History page.
8. THE System SHALL never block the HTTP response thread waiting for video processing to complete.

---

### Requirement 7: Face Detection and Matching Pipeline

**User Story:** As an investigator, I want the system to automatically detect faces in uploaded evidence and match them against the watchlist, so that I receive accurate identification results.

#### Acceptance Criteria

1. WHEN processing a Media file, THE System SHALL call `load_all_embeddings()` at the start of each job to retrieve the current Gallery for every Suspect with `enroll_status='ready'`.
2. WHEN processing an image, THE System SHALL apply CLAHE preprocessing, run face detection via the FaceEngine, and compare each detected face Embedding against all Suspect Galleries using Cosine Similarity.
3. WHEN processing a video, THE System SHALL sample frames at the configured skip interval, apply CLAHE preprocessing to each sampled frame, and run face detection on each sampled frame.
4. WHEN a face Embedding's maximum Cosine Similarity across all Suspect Galleries meets or exceeds the Similarity Threshold, THE System SHALL classify the face as a match for the Suspect with the highest score.
5. WHEN a match is found, THE System SHALL create a DetectionResult record containing the suspect ID, confidence score, frame number, timestamp in video, and paths to the matched frame crop and annotated frame.
6. WHEN a face is detected but no match meets the Similarity Threshold, THE System SHALL create an UnknownIdentity record containing the face crop path, the pickled Embedding, the best score, and the closest Suspect name.
7. WHEN processing a video, THE System SHALL draw a labelled bounding box with corner accents on each matched face and a grey bounding box on each unmatched face.
8. WHEN one or more Suspects are matched in a video frame, THE System SHALL draw a red alert overlay banner across the top of the frame for `ALERT_DURATION_FRAMES` subsequent frames.
9. WHEN video processing completes, THE System SHALL re-encode the annotated output using ffmpeg via `subprocess.run` to produce a browser-playable H.264 MP4 file.
10. THE System SHALL store the output video path as a relative path by stripping the static folder prefix before writing to the `output_path` column.
11. WHEN processing completes successfully, THE System SHALL set `status='done'`, `progress=100`, and `finished_at` on the Media record.
12. IF an unhandled exception occurs during processing, THEN THE System SHALL set `status='failed'` and store the exception message in `error_msg` on the Media record.
13. THE System SHALL update the `progress` column (0–100) at regular intervals during video processing so that the frontend polling endpoint can report live progress.

---

### Requirement 8: Results Viewer

**User Story:** As an investigator, I want to view the detection results for a specific evidence file, so that I can review identified suspects and annotated footage.

#### Acceptance Criteria

1. THE System SHALL render a results page at `/results/<media_id>` showing case metadata in a left sidebar and detection output in the main column.
2. WHEN `status='done'`, THE System SHALL display the Annotated Video in an HTML5 video player.
3. WHILE `status='running'`, THE System SHALL display a progress spinner and the current progress percentage.
4. WHEN `status='failed'`, THE System SHALL display the contents of `error_msg` in an error box.
5. THE System SHALL display a grid of DetectionResult cards, each showing the face crop image, Suspect name, timestamp, and a confidence score bar.
6. THE System SHALL embed JavaScript that polls `/job-status/<media_id>` every 3 seconds while the job is not in a terminal state (`done` or `failed`).
7. WHEN the polling response returns `status='done'`, THE System SHALL reload the page via `window.location.reload()` to display the final results.
8. THE System SHALL provide a link from the results page to the CaseReport editor for the associated Media record.

---

### Requirement 9: Job Status API

**User Story:** As a frontend client, I want a lightweight JSON endpoint for job status, so that the results page can poll for live progress without full page reloads.

#### Acceptance Criteria

1. THE System SHALL expose a `/job-status/<media_id>` route that returns a JSON object containing `status`, `progress`, `output_path`, and `error_msg`.
2. WHEN the requested `media_id` does not exist in the database, THE System SHALL return a JSON error response with HTTP status 404.
3. THE System SHALL return the response with `Content-Type: application/json`.

---

### Requirement 10: Evidence History

**User Story:** As an investigator, I want to browse all previously submitted evidence files, so that I can quickly locate and revisit any case.

#### Acceptance Criteria

1. THE System SHALL render a history page at `/history` listing all Media records ordered by upload date descending.
2. EACH Media card SHALL display the case ID, filename, inspector name, region, priority badge, upload date, job status badge, and a link to the results page.
3. THE System SHALL provide a live search input that filters cards by case ID, filename, inspector name, or region without a full page reload.
4. THE System SHALL provide a status filter control that limits displayed cards to a selected Job Status value.

---

### Requirement 11: Dashboard

**User Story:** As a supervisor, I want a dashboard with KPI cards and charts, so that I can monitor overall system activity at a glance.

#### Acceptance Criteria

1. THE System SHALL render a dashboard at `/dashboard` displaying KPI cards for total Suspect count, total Media count, processed Media count, and pending Media count.
2. THE System SHALL display a list of the five most recent DetectionResult records on the dashboard.
3. THE System SHALL render a bar chart of Media counts grouped by region using the HTML5 Canvas API without an external chart library.
4. THE System SHALL render a line chart of DetectionResult counts grouped by calendar month using the HTML5 Canvas API without an external chart library.
5. THE System SHALL animate KPI counter values from zero to their final value on page load.

---

### Requirement 12: Reports Page

**User Story:** As a supervisor, I want a reports page with aggregated statistics, so that I can assess detection performance across cases and regions.

#### Acceptance Criteria

1. THE System SHALL render a reports page at `/reports` displaying KPI totals for all-time detections, average confidence score, detections this week, and detections this month.
2. THE System SHALL render a priority distribution donut chart using the HTML5 Canvas API.
3. THE System SHALL render a monthly detection trend line chart using the HTML5 Canvas API.
4. THE System SHALL display a region breakdown table showing, for each region, the total Media count, total match count, and match rate percentage.
5. THE System SHALL display a recent detections feed showing the last 20 DetectionResult records with suspect name, case ID, confidence, and timestamp.

---

### Requirement 13: Case Report Editor

**User Story:** As an investigator, I want to compose and save a rich-text case report for each evidence submission, so that I can document findings in a structured, printable format.

#### Acceptance Criteria

1. THE System SHALL render a case report editor at `/report/custom/<media_id>` pre-populated with a default HTML template containing case metadata fields.
2. THE System SHALL embed a rich text editor (TinyMCE or equivalent) in the editor page.
3. WHEN a User saves the report via the editor, THE System SHALL send the HTML content via an AJAX POST request to `/report/custom/save/<media_id>`.
4. WHEN the save endpoint receives valid HTML content, THE System SHALL create or update the CaseReport record for the given Media and return a JSON success response.
5. IF the `media_id` does not exist, THEN THE System SHALL return a JSON error response with HTTP status 404.

---

### Requirement 14: Regions and Reported Cases Pages

**User Story:** As an investigator, I want to view a list of defined regions and filter reported cases by status and region, so that I can manage geographically distributed investigations.

#### Acceptance Criteria

1. THE System SHALL render a regions page at `/regions` displaying the static list of defined region names.
2. THE System SHALL render a reported cases page at `/reported_cases` listing all Media records that have an associated CaseReport.
3. THE System SHALL provide filter controls on the reported cases page to narrow results by Job Status and Region.

---

### Requirement 15: Frontend Theme and Accessibility

**User Story:** As a User, I want a dark-themed interface with a light mode toggle, so that I can work comfortably in low-light environments and switch themes as needed.

#### Acceptance Criteria

1. THE System SHALL apply a dark colour theme by default using CSS custom properties.
2. THE System SHALL provide a theme toggle control that switches between dark and light mode.
3. WHEN a User toggles the theme, THE System SHALL persist the selected theme in `localStorage` and apply it on subsequent page loads without a flash of the wrong theme.
4. THE System SHALL render a responsive grid layout that collapses to a single column on viewports narrower than 1024px.
5. THE System SHALL display flash messages for success, error, info, and warning events using a consistent styled component.
6. THE System SHALL provide a live search input on the History and Suspects pages that filters results client-side without a server round-trip.

---

### Requirement 16: Embedding Serialisation and Round-Trip Integrity

**User Story:** As a system developer, I want embeddings to be correctly serialised and deserialised, so that face matching produces consistent results across application restarts.

#### Acceptance Criteria

1. THE System SHALL serialise each Gallery as a `pickle.dumps()` byte string of a Python list of float32 numpy arrays before writing to the `gallery` column.
2. THE System SHALL deserialise each Gallery via `pickle.loads()` when loading embeddings for matching.
3. FOR ALL valid Gallery objects, serialising then deserialising SHALL produce a list of numpy arrays that are element-wise equal to the originals (round-trip property).
4. THE System SHALL serialise each UnknownIdentity Embedding as a `pickle.dumps()` byte string of a single float32 numpy array before writing to the `embedding` column.
5. FOR ALL valid Embedding objects, serialising then deserialising SHALL produce a numpy array that is element-wise equal to the original (round-trip property).

---

### Requirement 17: Database Integrity and Cascade Behaviour

**User Story:** As a system developer, I want the database schema to enforce referential integrity, so that deleting a Suspect or Media record does not leave orphaned child records.

#### Acceptance Criteria

1. THE System SHALL define foreign key relationships such that deleting a Suspect record cascade-deletes all associated DetectionResult records.
2. THE System SHALL define foreign key relationships such that deleting a Media record cascade-deletes all associated DetectionResult, UnknownIdentity, and CaseReport records.
3. THE System SHALL enforce a one-to-one relationship between Media and CaseReport at the application layer.
4. THE System SHALL use SQLAlchemy ORM models for all database interactions and SHALL NOT use raw SQL strings in route handlers.

---

### Requirement 18: File Storage and Path Management

**User Story:** As a system developer, I want all uploaded and generated files to be stored under the Flask static folder with relative paths in the database, so that files are consistently served and paths remain valid across deployments.

#### Acceptance Criteria

1. THE System SHALL save all uploaded evidence files to `static/uploads/evidence/`.
2. THE System SHALL save all suspect photos to `static/uploads/suspects/`.
3. THE System SHALL save all face crop images to `static/uploads/crops/`.
4. THE System SHALL save all annotated video output files to `static/uploads/output/`.
5. WHEN storing a file path in the database, THE System SHALL call `_to_relative()` to strip the static folder prefix, storing only the path relative to the static root.
6. WHEN serving a stored file path to the frontend, THE System SHALL prepend the static URL prefix to reconstruct the full URL.
