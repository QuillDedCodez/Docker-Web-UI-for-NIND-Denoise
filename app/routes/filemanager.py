"""
app/routes/filemanager.py
App 1 – File Manager API
Handles: tree, upload, delete, mkdir, download, storage info.
"""

import os
import io
import shutil
import zipfile
import threading
import time
import uuid

from flask import Blueprint, request, jsonify, send_file
from app.shared.utils import (
    INPUT_ROOT, OUTPUT_ROOT,
    build_tree, mirror_path, safe_join,
    storage_info, load_settings, is_locked, set_lock, fmt_bytes,
    rename_output_suffixes,
    broadcast,
)

fm = Blueprint("filemanager", __name__)

# ── Download job store ────────────────────────────────────────────────────────
_dl_jobs: dict = {}   # job_id -> {status, type, total, done, buf, name, cancel_evt}
_dl_lock = threading.Lock()

def _dl_collect(section, rel_files, rel_dirs, rel_path):
    """Return list of (tag, abs_path, arcname). tag='file' for single, 'zip' for multi."""
    root   = OUTPUT_ROOT if section == "output" else INPUT_ROOT
    pairs  = []
    if rel_path:
        target = safe_join(root, rel_path)
        if os.path.isfile(target):
            return [("file", target, os.path.basename(target))]
        for dp, _, fnames in os.walk(target):
            for fn in fnames:
                fp = os.path.join(dp, fn)
                pairs.append(("zip", fp, os.path.relpath(fp, target)))
        return pairs
    for rel in rel_files:
        fp = safe_join(root, rel)
        if os.path.isfile(fp):
            pairs.append(("zip", fp, rel))
    for rel in rel_dirs:
        fd = safe_join(root, rel)
        if os.path.isdir(fd):
            for dp, _, fnames in os.walk(fd):
                for fn in fnames:
                    fp = os.path.join(dp, fn)
                    pairs.append(("zip", fp, os.path.relpath(fp, root)))
    return pairs


# ── Tree ──────────────────────────────────────────────────────────────────────

@fm.get("/api/tree")
def api_tree():
    return jsonify({
        "input":   build_tree(INPUT_ROOT),
        "output":  build_tree(OUTPUT_ROOT),
        "storage": storage_info(),
        "locked":  is_locked(),
    })


# ── Storage ───────────────────────────────────────────────────────────────────

@fm.get("/api/storage")
def api_storage():
    s = storage_info()
    return jsonify({**s,
        "input_used_fmt":  fmt_bytes(s["input_used"]),
        "output_used_fmt": fmt_bytes(s["output_used"]),
        "disk_free_fmt":   fmt_bytes(s["disk_free"]),
        "disk_total_fmt":  fmt_bytes(s["disk_total"]),
    })


# ── Upload ────────────────────────────────────────────────────────────────────

@fm.post("/api/upload")
def api_upload():
    if is_locked():
        return jsonify({"error": "Worker is running. Uploads are locked."}), 423

    _req_start  = time.time()   # wall-clock start — used for elapsed_ms in broadcast
    dest_rel    = request.form.get("dest", "")
    batch_id    = request.form.get("batch_id", "")
    batch_total = int(request.form.get("batch_total", 0))
    files       = request.files.getlist("files")
    settings    = load_settings()
    saved       = []

    for f in files:
        filename = (f.filename or "").replace("\\", "/").lstrip("/")
        if not filename:
            continue
        target = safe_join(INPUT_ROOT, os.path.join(dest_rel, filename))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        f.save(target)

        rel_dir = os.path.join(dest_rel, os.path.dirname(filename)).replace("\\", "/").strip("/")
        if rel_dir:
            out_dir = mirror_path(rel_dir, settings["folder_suffix"], settings["file_suffix"])
            os.makedirs(safe_join(OUTPUT_ROOT, out_dir), exist_ok=True)

        saved.append(filename)

    # Broadcast per-file progress so remote clients can track the batch
    if batch_id and saved:
        # file_size used by remote observer for folder progress propagation
        saved_file = saved[0]
        full_path  = os.path.join(INPUT_ROOT, dest_rel, saved_file) if dest_rel else os.path.join(INPUT_ROOT, saved_file)
        file_size  = os.path.getsize(full_path) if os.path.isfile(full_path) else 0
        broadcast("upload_file_done", {
            "batch_id":    batch_id,
            "batch_total": batch_total,
            "file":        saved_file,
            "file_size":   file_size,
            "elapsed_ms":  round((time.time() - _req_start) * 1000),
            "saved_count": len(saved),
        })

    broadcast("tree_updated", {})
    return jsonify({"saved": len(saved), "files": saved})


@fm.post("/api/upload_batch_start")
def api_upload_batch_start():
    """Register a new upload batch so remote clients see it immediately."""
    data  = request.get_json()
    broadcast("upload_batch_start", {
        "batch_id":   data.get("batch_id", ""),
        "title":      data.get("title", "Uploading"),
        "file_count": data.get("file_count", 0),
        "files":      data.get("files", []),
    })
    return jsonify({"ok": True})


@fm.post("/api/upload_batch_cancel")
def api_upload_batch_cancel():
    data = request.get_json()
    broadcast("upload_batch_cancel", {
        "batch_id":      data.get("batch_id", ""),
        "pending_files": data.get("pending_files", []),  # for remote dual-fill
    })
    return jsonify({"ok": True})


@fm.post("/api/upload_file_cancel")
def api_upload_file_cancel():
    """Client cancelled an individual file — broadcast so other observers update."""
    data = request.get_json()
    broadcast("upload_file_cancel", {
        "batch_id":  data.get("batch_id", ""),
        "file":      data.get("file", ""),
        "file_size": data.get("file_size", 0),  # needed for dual-fill on remote bars
    })
    return jsonify({"ok": True})


@fm.post("/api/upload_cancel_all")
def api_upload_cancel_all():
    """Tab-close beacon: cancel every active batch the client was uploading.

    Called exclusively via navigator.sendBeacon on pagehide/beforeunload, so
    the request body is a JSON blob.  Broadcasts upload_batch_cancel for each
    batch_id so remote observers can mark the card as cancelled.
    """
    data = request.get_json(silent=True) or {}
    for bid in data.get("batch_ids", []):
        broadcast("upload_batch_cancel", {"batch_id": bid, "pending_files": []})
    return jsonify({"ok": True})


# ── Delete ────────────────────────────────────────────────────────────────────

@fm.post("/api/delete")
def api_delete():
    if is_locked():
        return jsonify({"error": "Worker is running. Deletions are locked."}), 423

    data     = request.get_json()
    rel_path = data.get("path", "")
    section  = data.get("section", "both")   # "input" | "output" | "both"
    settings = load_settings()

    def _rm(root, rel):
        target = safe_join(root, rel)
        if os.path.isdir(target):
            shutil.rmtree(target)
        elif os.path.isfile(target):
            os.remove(target)

    if section in ("input", "both"):
        _rm(INPUT_ROOT, rel_path)
    if section == "both":
        out_rel = mirror_path(rel_path, settings["folder_suffix"], settings["file_suffix"])
        _rm(OUTPUT_ROOT, out_rel)
        # Empty output folders are intentional mirrors of the input tree — never prune.
    elif section == "output":
        # Output-only delete: remove just the file, never touch folder structure.
        target = safe_join(OUTPUT_ROOT, rel_path)
        if os.path.isfile(target):
            os.remove(target)
        # Empty output folders are kept intact — they mirror the input tree structure.

    broadcast("tree_updated", {})   # notify all clients
    return jsonify({"ok": True})


# ── Rename output suffixes ────────────────────────────────────────────────────

@fm.post("/api/rename_suffixes")
def api_rename_suffixes():
    if is_locked():
        return jsonify({"error": "Worker is running."}), 423
    data               = request.get_json()
    old_folder_suffix  = data.get("old_folder_suffix",  "")
    new_folder_suffix  = data.get("new_folder_suffix",  "")
    old_file_suffix    = data.get("old_file_suffix",    "")
    new_file_suffix    = data.get("new_file_suffix",    "")
    result = rename_output_suffixes(
        old_folder_suffix, new_folder_suffix,
        old_file_suffix,   new_file_suffix,
    )
    broadcast("tree_updated", {})   # suffix rename changes tree appearance
    return jsonify(result)


# ── Mkdir ─────────────────────────────────────────────────────────────────────

@fm.post("/api/mkdir")
def api_mkdir():
    if is_locked():
        return jsonify({"error": "Worker is running."}), 423

    data     = request.get_json()
    rel      = data.get("path", "").strip("/")
    settings = load_settings()

    os.makedirs(safe_join(INPUT_ROOT,  rel), exist_ok=True)
    out_rel = mirror_path(rel, settings["folder_suffix"], settings["file_suffix"])
    os.makedirs(safe_join(OUTPUT_ROOT, out_rel), exist_ok=True)

    broadcast("tree_updated", {})   # notify all clients
    return jsonify({"ok": True})



# ── Download (new: prepare + stream result) ──────────────────────────────────

@fm.post("/api/download_prepare")
def api_download_prepare():
    """Start a background zip job; returns job_id immediately.
    For single files, marks ready immediately so the client just fetches the result."""
    data     = request.get_json()
    section  = data.get("section", "output")
    rel_path = data.get("path", "")          # whole-section or single path
    files    = data.get("files", [])
    dirs     = data.get("dirs",  [])
    zip_name = data.get("name", "download.zip")

    pairs = _dl_collect(section, files, dirs, rel_path)
    if not pairs:
        return jsonify({"error": "Nothing to download"}), 400

    job_id = uuid.uuid4().hex[:10]

    # Single file — no zip needed
    if len(pairs) == 1 and pairs[0][0] == "file":
        _, abs_path, name = pairs[0]
        with _dl_lock:
            _dl_jobs[job_id] = {
                "status": "ready", "type": "file",
                "abs_path": abs_path, "name": name,
                "total": 1, "done": 1, "cancel_evt": threading.Event(),
            }
        return jsonify({"job_id": job_id, "type": "file", "total": 1})

    total = len(pairs)
    cancel_evt = threading.Event()
    with _dl_lock:
        _dl_jobs[job_id] = {
            "status": "building", "type": "zip",
            "total": total, "done": 0,
            "buf": None, "name": zip_name,
            "cancel_evt": cancel_evt,
        }

    def _build():
        buf = io.BytesIO()
        try:
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, (_, abs_path, arcname) in enumerate(pairs):
                    if cancel_evt.is_set():
                        with _dl_lock:
                            _dl_jobs[job_id]["status"] = "cancelled"
                        return
                    zf.write(abs_path, arcname)
                    done = i + 1
                    pct  = round(done / total * 100)
                    with _dl_lock:
                        _dl_jobs[job_id]["done"] = done

            buf.seek(0)
            with _dl_lock:
                _dl_jobs[job_id]["status"] = "ready"
                _dl_jobs[job_id]["buf"]    = buf

        except Exception as e:
            with _dl_lock:
                _dl_jobs[job_id]["status"] = "error"


    threading.Thread(target=_build, daemon=True).start()
    return jsonify({"job_id": job_id, "type": "zip", "total": total})


@fm.get("/api/download_result/<job_id>")
def api_download_result(job_id):
    with _dl_lock:
        job = _dl_jobs.get(job_id)
    if not job or job["status"] != "ready":
        return jsonify({"error": "Not ready or not found"}), 404

    if job["type"] == "file":
        resp = send_file(
            job["abs_path"], as_attachment=True,
            download_name=job["name"],
            conditional=False,
        )
    else:
        buf = job["buf"]
        buf.seek(0)
        resp = send_file(
            buf, as_attachment=True,
            download_name=job["name"],
            mimetype="application/zip",
        )

    # Schedule cleanup after 5 minutes
    def _cleanup():
        time.sleep(300)
        with _dl_lock:
            _dl_jobs.pop(job_id, None)
    threading.Thread(target=_cleanup, daemon=True).start()
    return resp


@fm.post("/api/download_cancel/<job_id>")
def api_download_cancel(job_id):
    with _dl_lock:
        job = _dl_jobs.get(job_id)
    if job:
        job["cancel_evt"].set()
        job["status"] = "cancelled"
    return jsonify({"ok": True})


@fm.get("/api/download_status/<job_id>")
def api_download_status(job_id):
    with _dl_lock:
        job = _dl_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    total = job.get("total", 1) or 1
    done  = job.get("done",  0)
    return jsonify({
        "status": job["status"],
        "pct":    round(done / total * 100),
        "done":   done,
        "total":  total,
    })


# ── Force unlock ──────────────────────────────────────────────────────────────

@fm.post("/api/force_unlock")
def api_force_unlock():
    """
    Emergency endpoint — clears a stale lock.json left by a container crash.
    Only effective when no session is truly running.  Any restart of the
    container guarantees no live worker threads exist, so this is safe to call
    immediately after the container comes back up.
    """
    set_lock(False)
    broadcast("tree_updated", {})   # clients re-fetch tree (which carries locked=False)
    return jsonify({"ok": True})