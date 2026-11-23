"""Headless smoke test of predict_app.py; writes outputs/ml/gui_smoke_test.json.

Run with:  uv run python scripts/smoke_predict_app.py
1. Start `streamlit run predict_app.py --server.headless true` and check /_stcore/health.
2. Run the page with Streamlit's AppTest for every model in the menu (test-split image)
   and record any exception and the prediction text shown.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ml" / "gui_smoke_test.json"
PORT = 8599


def health_check() -> int:
    """Start the app headless, wait for the health endpoint, return its HTTP status."""
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ROOT / "predict_app.py"),
        "--server.headless",
        "true",
        "--server.port",
        str(PORT),
    ]
    env = {**os.environ, "YOLO_OFFLINE": "true"}
    server = subprocess.Popen(
        command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        for _ in range(60):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/_stcore/health") as reply:
                    return reply.status
            except OSError:
                time.sleep(1)
        return 0
    finally:
        server.terminate()
        server.wait(timeout=30)


def app_runs() -> list[dict]:
    """Run the page once per model and record what it shows."""
    results = []
    app = AppTest.from_file(str(ROOT / "predict_app.py"), default_timeout=300).run()
    models = app.selectbox[1].options  # selectbox 0 = test image, 1 = model
    for model in models:
        app.selectbox[1].set_value(model).run()
        texts = [m.value for m in app.markdown] + [e.value for e in app.error]
        texts += [w.value for w in app.warning] + [i.value for i in app.info]
        texts += [s.value for s in app.success]
        results.append(
            {
                "model": model,
                "exceptions": len(app.exception),
                "shown": [t for t in texts if "Prediction" in t or "CORRECT" in t or "Score" in t],
            }
        )
    return results


def main() -> None:
    status = health_check()
    runs = app_runs()
    report = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "health_status": status,
        "apptest_exceptions": sum(r["exceptions"] for r in runs),
        "models_tested": runs,
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
