"""Entry point to launch the AI INTERVIEW SIMULATOR Streamlit application cleanly."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Resolve project root
ROOT = Path(__file__).resolve().parent

# Ensure project root is in sys.path and PYTHONPATH
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

# Ensure ~/.streamlit/credentials.toml exists with empty email to prevent any first-run terminal prompt
try:
    streamlit_home = Path.home() / ".streamlit"
    streamlit_home.mkdir(parents=True, exist_ok=True)
    cred_file = streamlit_home / "credentials.toml"
    if not cred_file.exists():
        cred_file.write_text('[general]\nemail = ""\n', encoding="utf-8")
except Exception:
    pass


def main() -> int:
    """Launch the modern card-based Streamlit application without startup prompts."""
    app_path = ROOT / "frontend" / "app.py"
    if not app_path.exists():
        # Fallback check if in root with ai_interview_simulator subfolder
        alt_app_path = ROOT / "ai_interview_simulator" / "frontend" / "app.py"
        if alt_app_path.exists():
            app_path = alt_app_path

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--theme.primaryColor=#059669",
        "--theme.backgroundColor=#FFFFFF",
        "--theme.secondaryBackgroundColor=#F8FAFC",
        "--theme.textColor=#1E1B4B",
    ]
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
