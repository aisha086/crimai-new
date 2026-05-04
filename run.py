"""
CrimAI development server entry point.

Usage:
    python run.py
"""

from crimai.app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
