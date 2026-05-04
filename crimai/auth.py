"""
CrimAI authentication blueprint.

Routes
------
GET/POST /auth/register      – Registration form
GET/POST /auth/login         – Login form
GET      /auth/logout        – Invalidate session
GET/POST /auth/forgot-password – Placeholder forgot-password page

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 1.8
"""

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import login_required, login_user, logout_user
from werkzeug.security import check_password_hash, generate_password_hash

from crimai.models import User, db

auth = Blueprint('auth', __name__)


@auth.route('/register', methods=['GET', 'POST'])
def register():
    """Registration form: validate unique username, hash password, create User."""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            flash('Username and password are required.', 'error')
            return render_template('auth/register.html')

        # Check for duplicate username (Requirement 1.3)
        existing = User.query.filter_by(username=username).first()
        if existing:
            flash('Username already exists. Please choose a different one.', 'error')
            return render_template('auth/register.html')

        # Hash password with pbkdf2:sha256 (Requirements 1.2, 1.8)
        password_hash = generate_password_hash(password, method='pbkdf2:sha256')
        user = User(username=username, password_hash=password_hash)
        db.session.add(user)
        db.session.commit()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('auth.login'))

    return render_template('auth/register.html')


@auth.route('/login', methods=['GET', 'POST'])
def login():
    """Login form: validate credentials, create session."""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('main.dashboard'))

        # Requirement 1.5: flash error, do not create session
        flash('Invalid username or password.', 'error')

    return render_template('auth/login.html')


@auth.route('/logout')
@login_required
def logout():
    """Invalidate session and redirect to login (Requirement 1.7)."""
    logout_user()
    return redirect(url_for('auth.login'))


@auth.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Placeholder forgot password page."""
    if request.method == 'POST':
        flash(
            'Password reset functionality is not yet available. '
            'Please contact your administrator.',
            'info',
        )
    return render_template('auth/forgot_password.html')
