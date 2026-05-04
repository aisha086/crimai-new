"""
Property-based tests for the CrimAI Flask application.

This file contains Hypothesis property tests P3–P6.
Properties P1 and P2 will be added in task 15.2.

Each property is annotated with the requirement(s) it validates.
"""

from hypothesis import given, settings, assume
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Property 6: Password Hash Non-Reversibility
# Feature: crimai-flask-app, Property 6: Password Hash Non-Reversibility
# Validates: Requirements 1.2, 1.8
# ---------------------------------------------------------------------------

@settings(max_examples=20, deadline=None)
@given(
    password=st.text(min_size=1, max_size=50),
    other=st.text(min_size=1, max_size=50),
)
def test_password_hash_non_reversibility(password, other):
    """Property 6: password_hash != plaintext, check_password_hash is correct.

    For any plaintext password:
    1. The stored hash must not equal the plaintext
    2. check_password_hash(hash, password) must return True
    3. check_password_hash(hash, other) must return False when other != password

    **Validates: Requirements 1.2, 1.8**
    """
    from werkzeug.security import check_password_hash, generate_password_hash

    password_hash = generate_password_hash(password, method="pbkdf2:sha256")

    # Property: hash != plaintext
    assert password_hash != password

    # Property: check_password_hash returns True for the original password
    assert check_password_hash(password_hash, password) is True

    # Property: check_password_hash returns False for a different string
    assume(other != password)
    assert check_password_hash(password_hash, other) is False
