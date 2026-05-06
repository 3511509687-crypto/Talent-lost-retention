from __future__ import annotations

import os

from web_latest_app import app


if __name__ == "__main__":
    debug_enabled = str(os.environ.get("HR_WEB_DEBUG", "")).strip().lower() in {"1", "true", "yes", "on"}
    host = os.environ.get("HR_WEB_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.environ.get("HR_WEB_PORT", "5000"))
    app.run(host=host, port=port, debug=debug_enabled)
