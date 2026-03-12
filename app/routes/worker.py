"""
app/routes/worker.py
App 3 – Parallel Worker Engine
- Round-robin distribution of commands across N workers
- Each worker runs commands ONE-BY-ONE via subprocess
- Real-time SocketIO updates: progress, timing, VRAM per worker
- Overwrite / skip logic checked before each run
- Sets global lock while running; clears on completion or cancel
- Writes "running" status to commands.json when a command starts
- Saves last session's results to history.json on completion
"""

import os
import signal
import time
import threading
import subprocess

from flask import Blueprint, jsonify
from app.shared.utils import (
    load_commands, save_commands,
    load_settings, save_settings, set_lock, get_vram_info, get_ram_info,
    STATE_DIR, init_sio, broadcast,
    load_selection,
)

import json

wk = Blueprint("worker", __name__)

HISTORY_FILE = os.path.join(STATE_DIR, "commands_history.json")


def _save_history(cmds: list):
    """Persist the completed session's commands as a 1-step history."""
    with open(HISTORY_FILE, "w") as f:
        json.dump(cmds, f, indent=2)

def _load_history() -> list:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


# ── Session state (in-memory, reset on each run) ──────────────────────────────

# Hard cap — enforced server-side regardless of what the client sends.
MAX_WORKERS = 16

_session = {
    "running":       False,
    "cancelled":     False,
    "workers":       [],
    "start":         None,
    "worker_config": {},   # config used by the winning start: {num_workers, overwrite}
}
_sio_ref = None   # set by main.py after socketio is created

# ── Status-emit throttle ──────────────────────────────────────────────────────
# _build_status() calls get_vram_info() (pynvml) which is cheap but non-trivial.
# Firing it on every stdout line of a verbose script can spike pynvml polling.
# _emit_status_throttled() caps updates to _STATUS_EMIT_INTERVAL seconds in the
# hot readline loop; the unthrottled _emit() is still used at state transitions.
_last_status_emit   = 0.0
_STATUS_EMIT_INTERVAL = 0.25   # seconds — max ~4 worker-card refreshes per second


def init_socketio(sio):
    global _sio_ref
    _sio_ref = sio
    init_sio(sio)          # share with all routes via utils.broadcast()
    _register_sync(sio)    # /sync namespace — multi-client state sync
    _register_events(sio)  # /worker namespace — session control


def _emit(event, data):
    if _sio_ref:
        _sio_ref.emit(event, data, namespace="/worker")


def _emit_status_throttled():
    """Throttled variant for the hot stdout readline loop."""
    global _last_status_emit
    now = time.time()
    if now - _last_status_emit >= _STATUS_EMIT_INTERVAL:
        _last_status_emit = now
        _emit("worker_update", _build_status())


def _collect_all_commands() -> list:
    """Flatten all workers' command lists back into one list."""
    return [cmd for w in _session["workers"] for cmd in w["commands"]]


def _save_and_broadcast_commands() -> list:
    """Collect once, persist, and push to all clients — eliminates double calls."""
    cmds = _collect_all_commands()
    save_commands(cmds)
    broadcast("commands_updated", {"commands": cmds})
    return cmds


# ── Worker thread ─────────────────────────────────────────────────────────────

def _kill_proc(proc: subprocess.Popen) -> None:
    """
    Terminate a subprocess and its entire process group (covers PyTorch
    data-loader workers and any other children spawned by denoise_image.py).

    Strategy:
      1. SIGTERM to the process group — gives the script a chance to flush
         and close files cleanly.
      2. Wait up to 5 s.
      3. If still alive: SIGKILL to the process group — unconditional.
      4. Final wait to reap the zombie and release the pipe.
    """
    if proc is None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        pgid = None

    # Try graceful SIGTERM on the whole group first
    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
    except (ProcessLookupError, OSError):
        pass

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        # Still alive — force-kill
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGKILL)
            else:
                proc.kill()
        except (ProcessLookupError, OSError):
            pass
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass  # give up; OS will reap on container exit

    # Drain and close stdout pipe so no file-descriptor leak
    try:
        if proc.stdout and not proc.stdout.closed:
            proc.stdout.read()   # drain remaining buffer
            proc.stdout.close()
    except OSError:
        pass


def _worker_thread(worker_id: int, commands: list):
    w = _session["workers"][worker_id]
    w["status"]     = "running"
    w["start_time"] = time.time()
    w["last_time"]  = time.time()
    w["proc"]       = None   # active subprocess reference for immediate cancel

    for cmd_obj in commands:
        if _session["cancelled"]:
            w["status"] = "cancelled"
            break

        # Re-check overwrite/skip at runtime
        settings = load_settings()
        out_path = cmd_obj["output"]
        if not settings.get("overwrite", True) and os.path.exists(out_path):
            cmd_obj["status"] = "skipped"
            w["skipped"] += 1
            _save_and_broadcast_commands()
            _emit("worker_update", _build_status())
            continue

        cmd_obj["status"] = "running"
        w["current_file"]  = os.path.basename(cmd_obj["input"])
        w["current_start"] = time.time()
        _save_and_broadcast_commands()
        _emit("worker_update", _build_status())

        # When overwrite=True and the output file already exists, remove it
        # before spawning the script.  This makes the post-run existence check
        # (success = os.path.exists(out_path)) genuinely meaningful: it can
        # only be True if the script produced the file during THIS run.
        # Without this, a pre-existing file from a previous session would make
        # the check return True even if the script silently skipped writing
        # (denoise_image.py has its own internal guard) or failed outright —
        # causing the session report to falsely count the file as completed.
        if settings.get("overwrite", True) and os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError as rm_err:
                _emit("terminal_line", {
                    "worker_id": w["id"],
                    "text": f"[warn] could not remove existing output before overwrite: {rm_err}",
                })

        proc    = None
        success = False
        try:
            proc = subprocess.Popen(
                cmd_obj["cmd"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                # New process group so os.killpg covers all children
                # (PyTorch data-loader workers, any sub-scripts, etc.)
                start_new_session=True,
            )
            w["proc"] = proc   # expose for immediate cancel from on_cancel

            # Stream output line by line, check cancel on each line
            cancelled_mid = False
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    _emit("terminal_line", {"worker_id": w["id"], "text": line})
                _emit_status_throttled()   # rate-limited: max 4 updates/sec
                if _session["cancelled"]:
                    cancelled_mid = True
                    break   # stop reading; finally block will kill the process

            if cancelled_mid:
                # Log last returncode if the process somehow already exited
                _emit("terminal_line", {
                    "worker_id": w["id"],
                    "text": "— cancelled: terminating process group —",
                })
            else:
                # Natural exit — just reap the process
                proc.wait()
                rc = proc.returncode
                if rc not in (0, None):
                    _emit("terminal_line", {
                        "worker_id": w["id"],
                        "text": f"[exit {rc}] (non-zero exit — checking output file)",
                    })
                # Output file existence is the ground truth
                success = os.path.exists(out_path)

        except Exception as e:
            _emit("terminal_line", {"worker_id": w["id"], "text": f"ERROR: {e}"})

        finally:
            # Always clean up the process and its group, regardless of how we got here
            _kill_proc(proc)
            w["proc"] = None

        elapsed = time.time() - w["current_start"]
        w["times"].append(elapsed)
        w["prev_time"]  = elapsed
        w["last_time"]  = time.time()
        w["current_file"] = ""

        if _session["cancelled"]:
            cmd_obj["status"] = "cancelled"
            w["status"] = "cancelled"
        elif success:
            cmd_obj["status"] = "done"
            w["completed"] += 1
        else:
            cmd_obj["status"] = "failed"
            w["failed"] += 1

        _save_and_broadcast_commands()
        _emit("worker_update", _build_status())

        if _session["cancelled"]:
            break

    w["end_time"] = time.time()
    w["proc"]     = None
    if w["status"] not in ("cancelled", "error"):
        w["status"] = "done"

    _emit("worker_update", _build_status())
    _check_all_done()


def _check_all_done():
    terminal = {"done", "cancelled", "error"}
    if all(w["status"] in terminal for w in _session["workers"]):
        _session["running"]       = False
        _session["worker_config"] = {}   # clear — session is over
        set_lock(False)
        cmds = _collect_all_commands()   # collect once — used for both history and broadcast
        _save_history(cmds)
        broadcast("history_updated", {})
        broadcast("commands_updated", {"commands": cmds})
        _emit("session_complete", _build_report())


# ── Status / Report builders ──────────────────────────────────────────────────

def _build_status() -> dict:
    now  = time.time()
    vram = get_vram_info()
    workers_out = []
    for w in _session["workers"]:
        t   = w.get("start_time")
        e   = w.get("end_time")
        elapsed     = (e - t) if e else ((now - t) if t else 0)
        cur_elapsed = (now - w["current_start"]) if w.get("current_start") and not w.get("end_time") else 0
        avg = sum(w["times"]) / len(w["times"]) if w["times"] else 0
        workers_out.append({
            "id":           w["id"],
            "status":       w["status"],
            "total":        w["total"],
            "completed":    w["completed"],
            "failed":       w["failed"],
            "skipped":      w["skipped"],
            "elapsed":      round(elapsed, 1),
            "current_time": round(cur_elapsed, 1),
            "prev_time":    round(w.get("prev_time", 0), 2),
            "avg_time":     round(avg, 2),
            "current_file": w.get("current_file", ""),
        })
    return {
        "running":       _session["running"],
        "workers":       workers_out,
        "vram":          vram,
        "ram":           get_ram_info(),
        "max_workers":   MAX_WORKERS,
        "worker_config": _session["worker_config"],  # winning config — all tabs sync to this
    }


def _build_report() -> dict:
    now  = time.time()
    ws   = _session["workers"]
    sess_t  = now - (_session["start"] or now)
    total_c = sum(w["completed"] for w in ws)
    total_f = sum(w["failed"]    for w in ws)
    total_s = sum(w["skipped"]   for w in ws)
    all_t   = [t for w in ws for t in w["times"]]
    return {
        "session_time":    round(sess_t, 1),
        "total_completed": total_c,
        "total_failed":    total_f,
        "total_skipped":   total_s,
        "mean_time":       round(sum(all_t) / len(all_t), 2) if all_t else 0,
        "throughput":      round(sess_t / total_c, 2) if total_c else 0,
        "workers":         _build_status()["workers"],
        "vram":            get_vram_info(),
    }


# ── /sync namespace — multi-client real-time synchronisation ─────────────────

_sync_clients = 0


def _register_sync(sio):
    global _sync_clients

    @sio.on("connect", namespace="/sync")
    def on_sync_connect():
        global _sync_clients
        _sync_clients += 1
        # Push current state to the newly connected client immediately
        sio.emit("clients_count",   {"count": _sync_clients},  namespace="/sync")
        sio.emit("commands_updated", {"commands": load_commands()}, namespace="/sync")
        sio.emit("history_updated",  {},                        namespace="/sync")
        s = load_settings()
        sio.emit("settings_updated", s, namespace="/sync")
        sio.emit("selection_updated", load_selection(), namespace="/sync")
        # Also notify all others of the new count
        sio.emit("clients_count",   {"count": _sync_clients},  namespace="/sync")

    @sio.on("disconnect", namespace="/sync")
    def on_sync_disconnect():
        global _sync_clients
        _sync_clients = max(0, _sync_clients - 1)
        sio.emit("clients_count", {"count": _sync_clients}, namespace="/sync")


# ── /worker namespace — session control ──────────────────────────────────────

def _register_events(sio):

    @sio.on("connect", namespace="/worker")
    def on_connect(sid=None):
        # Emit current state only to the connecting client, not all clients.
        # _emit() broadcasts — use sio.emit with to=request.sid instead.
        try:
            from flask import request as _req
            _sid = _req.sid
        except Exception:
            _sid = sid
        if _sid:
            sio.emit("worker_update", _build_status(), to=_sid, namespace="/worker")
        else:
            _emit("worker_update", _build_status())

    @sio.on("start", namespace="/worker")
    def on_start(data):
        if _session["running"]:
            # Losing client in a race — push current state so it syncs
            # its UI (slot count, pill, worker_config controls) immediately.
            _emit("worker_update", _build_status())
            return

        num_workers = max(1, min(MAX_WORKERS, int(data.get("workers", 1))))
        overwrite   = bool(data.get("overwrite", True))

        commands = [c for c in load_commands() if c.get("status") in ("pending", None)]
        if not commands:
            _emit("error", {"msg": "No pending commands. Generate commands in the Selector tab first."})
            return

        # ── Atomic config save ────────────────────────────────────────────────
        # Save overwrite to settings.json BEFORE launching threads.
        # All threads call load_settings() per-file at runtime, so this is the
        # single write that every worker will observe.  Doing it here (inside
        # on_start, while the session lock is not yet set) removes the race
        # between the old separate fetch('/api/settings') and socket.emit('start').
        saved = save_settings({"overwrite": overwrite})

        # Broadcast the merged settings to ALL clients so their Settings tab
        # also reflects the winning config immediately.
        broadcast("settings_updated", saved)

        # Store the full winning config so _build_status() carries it in
        # every subsequent worker_update — remote tabs apply it on receipt.
        _session["worker_config"] = {
            "num_workers": num_workers,
            "overwrite":   overwrite,
        }

        # Round-robin distribution
        buckets = [[] for _ in range(num_workers)]
        for i, cmd in enumerate(commands):
            buckets[i % num_workers].append(cmd)

        _session["running"]   = True
        _session["cancelled"] = False
        _session["start"]     = time.time()
        _session["workers"]   = [
            {
                "id":            i + 1,
                "total":         len(buckets[i]),
                "completed":     0,
                "failed":        0,
                "skipped":       0,
                "times":         [],
                "start_time":    None,
                "end_time":      None,
                "last_time":     None,
                "current_start": None,
                "prev_time":     0,
                "current_file":  "",
                "status":        "queued",
                "commands":      buckets[i],
                "proc":          None,
            }
            for i in range(num_workers)
        ]

        set_lock(True)
        _emit("worker_update", _build_status())

        for i in range(num_workers):
            t = threading.Thread(target=_worker_thread, args=(i, buckets[i]), daemon=True)
            t.start()

    @sio.on("cancel", namespace="/worker")
    def on_cancel():
        _session["cancelled"] = True
        # Immediately signal every active subprocess rather than waiting for
        # each readline loop to notice the flag on the next line of output.
        for w in _session["workers"]:
            proc = w.get("proc")
            if proc is not None:
                threading.Thread(
                    target=_kill_proc, args=(proc,), daemon=True
                ).start()
        _emit("worker_update", _build_status())

    @sio.on("status", namespace="/worker")
    def on_status():
        _emit("worker_update", _build_status())


# ── REST fallbacks ────────────────────────────────────────────────────────────

@wk.get("/api/worker/status")
def api_status():
    return jsonify(_build_status())


@wk.get("/api/worker/report")
def api_report():
    return jsonify(_build_report())


@wk.get("/api/commands/history")
def api_commands_history():
    """Return the last completed session's commands (1-step history)."""
    return jsonify(_load_history())


@wk.delete("/api/commands/history")
def api_commands_history_clear():
    """Delete the saved history file."""
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return jsonify({"ok": True})