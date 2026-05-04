"""
Unit tests for the auth blueprint (crimai/auth.py).

Validates Requirements 1.2, 1.3, 1.4, 1.5, 1.6, 1.7:
  1.2 - Registration hashes the password and persists a User record.
  1.3 - Duplicate username returns an error without creating a duplicate.
  1.4 - Valid credentials create an authenticated session and redirect to dashboard.
  1.5 - Invalid credentials do not create a session.
  1.6 - Unauthenticated request to a protected route redirects to /auth/login.
  1.7 - Logout invalidates the session and redirects to /auth/login.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from flask import Blueprint, Flask
from flask_login import login_required
from werkzeug.security import generate_password_hash

from crimai.auth import auth
from crimai.models import User, db, login_manager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    """Minimal Flask app with in-memory SQLite, auth blueprint, and a stub main blueprint."""
    _app = Flask(__name__, template_folder=None)
    _app.config["TESTING"] = True
    _app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite://"
    _app.config["SECRET_KEY"] = "test-secret"
    _app.config["WTF_CSRF_ENABLED"] = False

    db.init_app(_app)
    login_manager.init_app(_app)
    login_manager.login_view = "auth.login"

    _app.register_blueprint(auth, url_prefix="/auth")

    # Minimal main blueprint stub — provides the redirect targets used by auth routes
    main = Blueprint("main", __name__)

    @main.route("/dashboard")
    def dashboard():
        return "dashboard", 200

    @main.route("/protected")
    @login_required
    def protected():
        return "protected", 200

    _app.register_blueprint(main, url_prefix="/")

    with _app.app_context():
        db.create_all()
        yield _app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    """Flask test client with session cookies enabled."""
    return app.test_client()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _register(client, username: str = "officer1", password: str = "secret123"):
    """POST to /auth/register and return the response."""
    with patch("crimai.auth.render_template", return_value="<html></html>"):
        return client.post(
            "/auth/register",
            data={"username": username, "password": password},
            follow_redirects=False,
        )


def _login(client, username: str = "officer1", password: str = "secret123"):
    """POST to /auth/login and return the response."""
    with patch("crimai.auth.render_template", return_value="<html></html>"):
        return client.post(
            "/auth/login",
            data={"username": username, "password": password},
            follow_redirects=False,
        )


def _create_user(app, username: str = "officer1", password: str = "secret123") -> int:
    """Directly insert a User record and return its id."""
    with app.app_context():
        user = User(
            username=username,
            password_hash=generate_password_hash(password, method="pbkdf2:sha256"),
        )
        db.session.add(user)
        db.session.commit()
        return user.id


# ---------------------------------------------------------------------------
# Requirement 1.2 / 1.3 — Registration
# ---------------------------------------------------------------------------

class TestRegister:
    """Tests for POST /auth/register."""

    def test_register_unique_username_redirects_to_login(self, client):
        """Successful registration redirects to /auth/login (Req 1.2).

        Validates: Requirements 1.2
        """
        response = _register(client)
        assert response.status_code == 302
        assert "/auth/login" in response.headers["Location"]

    def test_register_unique_username_creates_user_record(self, app, client):
        """Successful registration persists exactly one User record (Req 1.2).

        Validates: Requirements 1.2
        """
        _register(client, username="newuser", password="pass1234")

        with app.app_context():
            users = User.query.filter_by(username="newuser").all()
            assert len(users) == 1

    def test_register_stores_hashed_password_not_plaintext(self, app, client):
        """The stored password_hash must not equal the plaintext password (Req 1.2, 1.8).

        Validates: Requirements 1.2
        """
        plaintext = "mysecretpassword"
        _register(client, username="hashtest", password=plaintext)

        with app.app_context():
            user = User.query.filter_by(username="hashtest").first()
            assert user is not None
            assert user.password_hash != plaintext

    def test_register_duplicate_username_returns_200(self, app, client):
        """Duplicate username re-renders the form (HTTP 200) (Req 1.3).

        Validates: Requirements 1.3
        """
        _create_user(app, username="existing", password="pass1")

        with patch("crimai.auth.render_template", return_value="<html></html>"):
            response = client.post(
                "/auth/register",
                data={"username": "existing", "password": "pass2"},
                follow_redirects=False,
            )
        assert response.status_code == 200

    def test_register_duplicate_username_does_not_create_duplicate(self, app, client):
        """Duplicate username registration must not insert a second User row (Req 1.3).

        Validates: Requirements 1.3
        """
        _create_user(app, username="dupuser", password="pass1")

        with patch("crimai.auth.render_template", return_value="<html></html>"):
            client.post(
                "/auth/register",
                data={"username": "dupuser", "password": "pass2"},
                follow_redirects=False,
            )

        with app.app_context():
            count = User.query.filter_by(username="dupuser").count()
            assert count == 1, f"Expected 1 user record, found {count}"


# ---------------------------------------------------------------------------
# Requirement 1.4 / 1.5 — Login
# ---------------------------------------------------------------------------

class TestLogin:
    """Tests for POST /auth/login."""

    def test_login_valid_credentials_redirects_to_dashboard(self, app, client):
        """Valid credentials redirect to /dashboard (Req 1.4).

        Validates: Requirements 1.4
        """
        _create_user(app, username="officer1", password="secret123")
        response = _login(client, username="officer1", password="secret123")

        assert response.status_code == 302
        assert "/dashboard" in response.headers["Location"]

    def test_login_valid_credentials_creates_session(self, app, client):
        """Valid credentials establish an authenticated session (Req 1.4).

        After login, a request to a @login_required route must succeed (200)
        rather than redirect to the login page.

        Validates: Requirements 1.4
        """
        _create_user(app, username="officer2", password="pass9876")

        with client.session_transaction() as sess:
            # Confirm no session before login
            assert "_user_id" not in sess

        _login(client, username="officer2", password="pass9876")

        # After login, the protected route should be accessible
        response = client.get("/protected")
        assert response.status_code == 200

    def test_login_invalid_password_returns_200(self, app, client):
        """Invalid password re-renders the login form (HTTP 200) (Req 1.5).

        Validates: Requirements 1.5
        """
        _create_user(app, username="officer3", password="correctpass")

        with patch("crimai.auth.render_template", return_value="<html></html>"):
            response = client.post(
                "/auth/login",
                data={"username": "officer3", "password": "wrongpass"},
                follow_redirects=False,
            )
        assert response.status_code == 200

    def test_login_invalid_credentials_does_not_create_session(self, app, client):
        """Invalid credentials must not establish an authenticated session (Req 1.5).

        After a failed login, a request to a @login_required route must
        redirect to the login page rather than succeed.

        Validates: Requirements 1.5
        """
        _create_user(app, username="officer4", password="realpass")

        with patch("crimai.auth.render_template", return_value="<html></html>"):
            client.post(
                "/auth/login",
                data={"username": "officer4", "password": "badpass"},
                follow_redirects=False,
            )

        # Protected route should redirect to login — no session was created
        response = client.get("/protected", follow_redirects=False)
        assert response.status_code == 302
        assert "/auth/login" in response.headers["Location"]

    def test_login_unknown_username_does_not_create_session(self, client):
        """Unknown username must not establish a session (Req 1.5).

        Validates: Requirements 1.5
        """
        with patch("crimai.auth.render_template", return_value="<html></html>"):
            client.post(
                "/auth/login",
                data={"username": "nobody", "password": "anything"},
                follow_redirects=False,
            )

        response = client.get("/protected", follow_redirects=False)
        assert response.status_code == 302
        assert "/auth/login" in response.headers["Location"]


# ---------------------------------------------------------------------------
# Requirement 1.7 — Logout
# ---------------------------------------------------------------------------

class TestLogout:
    """Tests for GET /auth/logout."""

    def _login_user(self, app, client, username="logoutuser", password="logoutpass"):
        """Helper: create a user and log in via the test client."""
        _create_user(app, username=username, password=password)
        _login(client, username=username, password=password)

    def test_logout_redirects_to_login(self, app, client):
        """Logout redirects to /auth/login (Req 1.7).

        Validates: Requirements 1.7
        """
        self._login_user(app, client)
        response = client.get("/auth/logout", follow_redirects=False)

        assert response.status_code == 302
        assert "/auth/login" in response.headers["Location"]

    def test_logout_invalidates_session(self, app, client):
        """After logout, a request to a protected route redirects to login (Req 1.7).

        Validates: Requirements 1.7
        """
        self._login_user(app, client, username="logoutuser2", password="pass")

        # Confirm the session is active before logout
        response_before = client.get("/protected")
        assert response_before.status_code == 200

        # Logout
        client.get("/auth/logout")

        # Protected route must now redirect to login
        response_after = client.get("/protected", follow_redirects=False)
        assert response_after.status_code == 302
        assert "/auth/login" in response_after.headers["Location"]


# ---------------------------------------------------------------------------
# Requirement 1.6 — Unauthenticated access to protected routes
# ---------------------------------------------------------------------------

class TestUnauthenticatedAccess:
    """Tests for unauthenticated requests to @login_required routes."""

    def test_unauthenticated_request_redirects_to_login(self, client):
        """An unauthenticated GET to a protected route redirects to /auth/login (Req 1.6).

        Validates: Requirements 1.6
        """
        response = client.get("/protected", follow_redirects=False)

        assert response.status_code == 302
        assert "/auth/login" in response.headers["Location"]

    def test_unauthenticated_request_does_not_return_200(self, client):
        """An unauthenticated request must not return the protected content (Req 1.6).

        Validates: Requirements 1.6
        """
        response = client.get("/protected", follow_redirects=False)
        assert response.status_code != 200
