"""Launch the FastAPI dashboard backend with uvicorn.

Usage:
    python scripts/start_dashboard.py --host 127.0.0.1 --port 8082
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run("dashboard.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
