"""
app/main.py
Single entry point — Flask + SocketIO serving all 3 apps on port 10010.
"""

import os
from flask import Flask
from flask_socketio import SocketIO

from app.routes.filemanager import fm
from app.routes.selector    import sel
from app.routes.worker      import wk, init_socketio

# ── App setup ─────────────────────────────────────────────────────────────────

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

app = Flask(__name__, template_folder=_TEMPLATE_DIR)
app.config["SECRET_KEY"] = "nind_denoise_secret"
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024 * 1024  # 10 GB upload limit

sio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── Register blueprints ───────────────────────────────────────────────────────

app.register_blueprint(fm)
app.register_blueprint(sel)
app.register_blueprint(wk)

# ── Wire up SocketIO to worker ────────────────────────────────────────────────

init_socketio(sio)

# ── Serve single-page UI ──────────────────────────────────────────────────────

@app.route("/")
def index():
    with open(os.path.join(_TEMPLATE_DIR, "index.html")) as f:
        return f.read()

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", 10010))
    sio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)