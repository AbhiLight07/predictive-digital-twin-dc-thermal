"""Launch the local 3D Predictive Digital Twin FastAPI server."""

from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from ttwin.twin3d.server import Twin3DService, create_app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 3D Predictive Digital Twin server.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser window")
    args = parser.parse_args()

    service = Twin3DService()
    print("Preloading D1/D2 TwinEngine models and forecast caches...")
    service.ensure_all()
    application = create_app(service=service)

    url = f"http://{args.host}:{args.port}"
    print(f"Twin3D server ready at {url}")

    if not args.no_browser:
        def _open_when_ready() -> None:
            time.sleep(1.0)
            webbrowser.open(url)
        threading.Thread(target=_open_when_ready, daemon=True).start()

    uvicorn.run(application, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
