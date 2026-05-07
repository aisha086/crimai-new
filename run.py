"""
CrimAI development server entry point.

Usage:
    python run.py

Set DATABASE_URL in .env or as an environment variable to use Supabase.
Leave it unset to use local SQLite.
"""

# Load .env before anything else
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — rely on shell environment

from crimai.app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
