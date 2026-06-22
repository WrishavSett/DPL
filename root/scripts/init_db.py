"""
One-off DB bootstrap: creates the SQLite schema if it doesn't exist yet.

Usage: python scripts/init_db.py

Not strictly required for normal operation — app/main.py's startup also
calls init_db() automatically — but useful for provisioning a fresh
database file ahead of time, e.g. in a deployment script or CI step.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python scripts/init_db.py` from the project root
# without installing the project as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.database import init_db  # noqa: E402


def main() -> None:
    init_db()
    print(f"Database ready at {settings.db_path}")


if __name__ == "__main__":
    main()