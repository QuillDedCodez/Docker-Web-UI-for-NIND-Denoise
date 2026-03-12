"""
app/routes/selector.py
App 2 – Selector & Command Generator API
Handles: settings, model discovery, command generation, selection persistence,
and model folder overrides.
Files with unsupported extensions are counted and categorised in skip reasons.
"""

import os
from flask import Blueprint, request, jsonify

from app.shared.utils import (
    INPUT_ROOT, OUTPUT_ROOT,
    load_settings, save_settings,
    discover_models, build_command, save_commands, load_commands,
    mirror_path, safe_join, is_locked, SUPPORTED_EXTS,
    LOCKED_NETWORKS, broadcast, STATE_DIR,
    load_selection, save_selection,
    apply_folder_override, clear_folder_override,
)

sel = Blueprint("selector", __name__)


# ── Settings ──────────────────────────────────────────────────────────────────

@sel.get("/api/settings")
def api_settings_get():
    return jsonify(load_settings())


@sel.post("/api/settings")
def api_settings_post():
    data = request.get_json()
    # Client must not be able to override which networks are locked
    data.pop("locked_networks", None)
    # If the active network was removed, fall back to the first locked network
    networks = data.get("networks", LOCKED_NETWORKS[:])
    if data.get("network") and data["network"] not in networks:
        data["network"] = LOCKED_NETWORKS[0]
    s = save_settings(data)
    broadcast("settings_updated", s)   # push to all clients
    return jsonify(s)


# ── Models ────────────────────────────────────────────────────────────────────

@sel.get("/api/models")
def api_models():
    return jsonify(discover_models())


# ── Generate commands from selection ─────────────────────────────────────────

@sel.post("/api/generate")
def api_generate():
    """
    Body: { "selected": ["rel/path/file.tiff", "rel/other.tiff", ...] }
    - Only files whose extension is in SUPPORTED_EXTS produce commands.
    - Skips are categorised into three buckets returned separately so the
      client can show an accurate warning rather than always blaming format.
    - Commands are saved to the shared state file and broadcast to all clients.
    """
    data     = request.get_json()
    selected = data.get("selected", [])
    settings = load_settings()

    skip_format  = []   # wrong / unsupported extension
    skip_missing = []   # extension OK but file not on disk
    skip_exists  = []   # overwrite=False and output already present
    commands     = []

    for rel in selected:
        ext = os.path.splitext(rel)[1].lower()
        if ext not in SUPPORTED_EXTS:
            skip_format.append(rel)
            continue

        in_abs = safe_join(INPUT_ROOT, rel)
        if not os.path.isfile(in_abs):
            skip_missing.append(rel)
            continue

        out_rel = mirror_path(rel, settings["folder_suffix"], settings["file_suffix"])
        out_abs = safe_join(OUTPUT_ROOT, out_rel)
        os.makedirs(os.path.dirname(out_abs), exist_ok=True)

        if not settings.get("overwrite", True) and os.path.exists(out_abs):
            skip_exists.append(rel)
            continue

        commands.append(build_command(in_abs, out_abs, settings))

    save_commands(commands)
    broadcast("commands_updated", {"commands": commands})

    skipped = skip_format + skip_missing + skip_exists
    return jsonify({
        "count":         len(commands),
        "skipped":       len(skipped),
        "skipped_files": skipped,
        "skip_format":   len(skip_format),
        "skip_missing":  len(skip_missing),
        "skip_exists":   len(skip_exists),
        "commands":      commands,
    })


# ── Read saved commands ───────────────────────────────────────────────────────

@sel.get("/api/commands")
def api_commands_get():
    return jsonify(load_commands())


@sel.post("/api/commands/clear")
def api_commands_clear():
    if is_locked():
        return jsonify({"error": "Worker is running."}), 423
    save_commands([])
    broadcast("commands_updated", {"commands": []})   # push to all clients
    return jsonify({"ok": True})

# ── Selector selection persistence ────────────────────────────────────────────

@sel.get("/api/selection")
def api_selection_get():
    return jsonify(load_selection())


@sel.post("/api/selection")
def api_selection_post():
    data = request.get_json()
    s    = save_selection(data)
    broadcast("selection_updated", s)
    return jsonify(s)


# ── Model folder overrides ────────────────────────────────────────────────────
# These routes were previously unregistered (apply_folder_override and
# clear_folder_override existed in utils.py but had no Flask endpoints).

@sel.post("/api/model_overrides/apply")
def api_model_override_apply():
    if is_locked():
        return jsonify({"error": "Worker is running."}), 423
    data = request.get_json()
    original_rel  = (data.get("original_rel")  or "").strip()
    override_name = (data.get("override_name") or "").strip()
    if not original_rel or not override_name:
        return jsonify({"error": "original_rel and override_name are required"}), 400
    result = apply_folder_override(original_rel, override_name)
    if "error" in result:
        return jsonify(result), 400
    broadcast("tree_updated", {})   # model list changed — clients re-fetch
    return jsonify(result)


@sel.post("/api/model_overrides/clear")
def api_model_override_clear():
    if is_locked():
        return jsonify({"error": "Worker is running."}), 423
    data = request.get_json()
    original_rel = (data.get("original_rel") or "").strip()
    if not original_rel:
        return jsonify({"error": "original_rel is required"}), 400
    result = clear_folder_override(original_rel)
    if "error" in result:
        return jsonify(result), 400
    broadcast("tree_updated", {})   # model list changed — clients re-fetch
    return jsonify(result)