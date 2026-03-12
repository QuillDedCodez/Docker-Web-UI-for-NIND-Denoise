"""
app/shared/utils.py
Shared state, filesystem helpers, path mirroring, VRAM monitoring.
"""

import os
import re
import json
import pathlib
import shutil

# ── Environment ───────────────────────────────────────────────────────────────

INPUT_ROOT  = os.environ.get("INPUT_ROOT",  "/data/input")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "/data/output")
MODEL_ROOT  = os.environ.get("MODEL_ROOT",  "/app/nind-denoise/models/nind denoise")
STATE_DIR   = os.environ.get("STATE_DIR",   "/app/state")
NIND_SCRIPT = os.environ.get("NIND_SCRIPT", "/app/nind-denoise/src/nind_denoise/denoise_image.py")

os.makedirs(INPUT_ROOT,  exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(STATE_DIR,   exist_ok=True)

SETTINGS_FILE  = os.path.join(STATE_DIR, "settings.json")
COMMANDS_FILE  = os.path.join(STATE_DIR, "commands.json")
LOCK_FILE      = os.path.join(STATE_DIR, "lock.json")
OVERRIDES_FILE = os.path.join(STATE_DIR, "folder_overrides.json")

# ── JSON file helpers ─────────────────────────────────────────────────────────

def _load_json(path: str, default):
    """Load JSON from path; return default if the file is missing or unreadable."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default

def _write_json(path: str, data, indent: int | None = None):
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)

# ── Locked-network tile defaults ──────────────────────────────────────────────
#
# To add or change a locked network, edit ONLY these three lines.
# Index N in each array corresponds to index N in the networks list.
# The rest of the codebase derives everything from NETWORK_TILE_DEFAULTS —
# no other file needs to be touched.
#
_LOCKED_NETWORK_NAMES: list[str] = ["UNet",  "UtNet"]
_LOCKED_NETWORK_CS:    list[int] = [440,     504    ]
_LOCKED_NETWORK_UCS:   list[int] = [320,     480    ]

# Runtime lookup dict built from the parallel arrays above.
# All code uses .get(name, fallback) — never indexes the raw arrays directly.
# Exposed to the client via _attach_derived() so the tile-size UI and
# build_command() both read from the same authoritative source.
NETWORK_TILE_DEFAULTS: dict[str, dict] = {
    name: {"cs": cs, "ucs": ucs}
    for name, cs, ucs in zip(
        _LOCKED_NETWORK_NAMES,
        _LOCKED_NETWORK_CS,
        _LOCKED_NETWORK_UCS,
    )
}

# Convenience: the first locked network is the application default.
_DEFAULT_NETWORK: str = _LOCKED_NETWORK_NAMES[0]

# ── Default settings ──────────────────────────────────────────────────────────
#
# Keys marked "read-only / never persisted" are architectural constants that are
# stripped from settings.json on every save and re-attached on every load.
# This means they always reflect the source-code values, not whatever a client
# last POSTed.  Edit the parallel arrays above — not the values here.
#
DEFAULT_SETTINGS = {
    "folder_suffix":    "_nind",
    "file_suffix":      "_nindimg",
    # Active network — initialised to the first locked network.
    "network":          _DEFAULT_NETWORK,
    # Locked networks list — derived from the parallel arrays above so the order
    # here is always in sync with NETWORK_TILE_DEFAULTS.
    "networks":         list(_LOCKED_NETWORK_NAMES),
    "use_tile_size":    False,      # if True, --cs / --ucs are appended to commands
    # Active tile sizes — initialised to the default network's architectural values.
    # These are the user's live session values and may differ from the defaults
    # once the user overrides them in the UI.
    "cs":               NETWORK_TILE_DEFAULTS[_DEFAULT_NETWORK]["cs"],
    "ucs":              NETWORK_TILE_DEFAULTS[_DEFAULT_NETWORK]["ucs"],
    "model_path":       "",
    "device":           "0",        # "0" = GPU, "-1" = CPU
    "workers":          1,
    "overwrite":        True,
    # ── Advanced / optional denoise_image.py arguments ────────────────────────
    # Empty string / None / False means the argument is omitted from commands.
    "overlap":          "",     # -ol  int   (script default: 6)
    "batch_size":       "",     # -b   int   (script default: 1; >1 broken upstream)
    "exif_method":      "",     # str  piexif | exiftool | noexif (script default: piexif)
    "model_parameters": "",     # str  "key=val,key=val"
    "max_subpixels":    "",     # int  abort if image exceeds this sub-pixel count
    "whole_image":      False,  # flag ignore cs/ucs, denoise whole image at once
    "pad":              "",     # int  padding per side (whole_image mode only)
    "debug":            False,  # flag store intermediate crops, verbose output
    # ── Architectural constants (read-only / never persisted) ─────────────────
    # supported_exts: the list the Selector uses to decide which files produce
    # commands.  Stored as a list so it survives JSON round-trips unchanged.
    # Formats nind-denoise can typically handle:
    #   TIFF:       ".tif", ".tiff"
    #   PNG:        ".png"
    #   JPEG:       ".jpg", ".jpeg"
    #   Camera RAW: ".cr2", ".cr3", ".nef", ".arw", ".dng"
    "supported_exts":   [".tif", ".tiff", ".png", ".jpg", ".jpeg"],
}

# ── Derived module-level constants ────────────────────────────────────────────

# Networks listed in _LOCKED_NETWORK_NAMES are locked — always present in the UI
# and cannot be removed by the user.  Derived here so LOCKED_NETWORKS is always
# consistent with NETWORK_TILE_DEFAULTS without any manual duplication.
LOCKED_NETWORKS: list[str] = list(_LOCKED_NETWORK_NAMES)

# Fast O(1) membership set for extension checks throughout the codebase.
# Derived from DEFAULT_SETTINGS["supported_exts"] — edit the list above, not here.
SUPPORTED_EXTS: set[str] = set(DEFAULT_SETTINGS["supported_exts"])

# ── Settings ──────────────────────────────────────────────────────────────────

# Keys that are architectural constants — derived from source code, never
# written to settings.json, but always re-attached on every load/save return.
_DERIVED_KEYS = ("locked_networks", "network_tile_defaults", "supported_exts")


def _ensure_locked_networks(s: dict) -> None:
    """In-place: guarantee every locked network is present at the front of s['networks']."""
    for n in LOCKED_NETWORKS:
        if n not in s.get("networks", []):
            s.setdefault("networks", [])
            s["networks"].insert(0, n)


def _attach_derived(s: dict) -> dict:
    """Attach all read-only derived keys to a settings dict before returning to callers."""
    s["locked_networks"]       = LOCKED_NETWORKS
    s["network_tile_defaults"] = NETWORK_TILE_DEFAULTS
    s["supported_exts"]        = sorted(SUPPORTED_EXTS)
    return s


def load_settings() -> dict:
    s = _load_json(SETTINGS_FILE, None)
    if s is not None:
        _ensure_locked_networks(s)
        return _attach_derived(s)
    d = dict(DEFAULT_SETTINGS)
    d["networks"] = list(dict.fromkeys(LOCKED_NETWORKS + d.get("networks", [])))
    return _attach_derived(d)


def save_settings(data: dict) -> dict:
    s = load_settings()
    s.update(data)
    _ensure_locked_networks(s)
    # Strip derived keys — never persisted; they are rebuilt from source on load
    for k in _DERIVED_KEYS:
        s.pop(k, None)
    _write_json(SETTINGS_FILE, s, indent=2)
    # Re-attach for the return value (API response)
    return _attach_derived(s)

# ── Lock (worker running) ─────────────────────────────────────────────────────

def set_lock(locked: bool):
    _write_json(LOCK_FILE, {"locked": locked})

def is_locked() -> bool:
    return _load_json(LOCK_FILE, {}).get("locked", False)

# ── Commands ──────────────────────────────────────────────────────────────────

def load_commands() -> list:
    return _load_json(COMMANDS_FILE, [])

def save_commands(cmds: list):
    _write_json(COMMANDS_FILE, cmds, indent=2)


# ── Selection persistence (shared between selector.py and worker.py) ──────────

SELECTION_FILE = os.path.join(STATE_DIR, "selection.json")


def load_selection() -> dict:
    return _load_json(SELECTION_FILE, {"files": [], "dirs": []})

def save_selection(data: dict) -> dict:
    s = {"files": list(data.get("files", [])), "dirs": list(data.get("dirs", []))}
    _write_json(SELECTION_FILE, s)
    return s


# Module-level constant — avoids rebuilding the list on every build_command call.
# Each tuple is (settings_key, cli_flag).  The value is only appended when
# non-empty after stripping, so omitting a key from settings silences that flag.
_OPTIONAL_SCALAR_ARGS: list[tuple[str, str]] = [
    ("overlap",          "--overlap"),
    ("batch_size",       "--batch_size"),
    ("exif_method",      "--exif_method"),
    ("model_parameters", "--model_parameters"),
    ("max_subpixels",    "--max_subpixels"),
]


def build_command(input_abs: str, output_abs: str, settings: dict) -> dict:
    active_net  = settings.get("network", "UNet")
    net_defs    = settings.get("network_tile_defaults", NETWORK_TILE_DEFAULTS).get(
                      active_net, {"cs": 440, "ucs": 320}
                  )

    cmd = ["python3", NIND_SCRIPT,
           "--network", active_net,
           "--device",  str(settings.get("device", "0"))]

    if settings.get("use_tile_size", False):
        cmd += ["--cs",  str(settings.get("cs",  net_defs["cs"])),
                "--ucs", str(settings.get("ucs", net_defs["ucs"]))]

    cmd += ["-i", input_abs, "-o", output_abs]

    if settings.get("model_path", ""):
        cmd += ["--model_path", settings["model_path"]]

    # ── Optional scalar arguments — data-driven; only added when non-empty ────
    for key, flag in _OPTIONAL_SCALAR_ARGS:
        val = settings.get(key, "")
        if str(val).strip():
            cmd += [flag, str(val)]

    if settings.get("whole_image", False):
        cmd += ["--whole_image"]
        pad = settings.get("pad", "")
        if str(pad).strip():
            cmd += ["--pad", str(pad)]

    if settings.get("debug", False):
        cmd += ["--debug"]

    return {
        "input":   input_abs,
        "output":  output_abs,
        "cmd":     cmd,
        "cmdstr":  " ".join(f'"{p}"' if " " in str(p) else str(p) for p in cmd),
        "status":  "pending",
    }


def safe_join(base: str, rel: str) -> str:
    target = os.path.realpath(os.path.join(base, rel))
    if not target.startswith(os.path.realpath(base)):
        raise ValueError("Path traversal blocked")
    return target


def mirror_path(rel: str, folder_suffix: str, file_suffix: str) -> str:
    """
    Convert an input-relative path to its mirrored output path.
    - Each folder segment gets folder_suffix appended.
    - Supported image file stems get file_suffix appended (see SUPPORTED_EXTS).
    - Other file types are passed through unchanged.
    """
    parts = pathlib.PurePosixPath(rel).parts
    new_parts = []
    for i, part in enumerate(parts):
        is_last = (i == len(parts) - 1)
        ext = os.path.splitext(part)[1].lower() if is_last else ""
        if is_last and ext in SUPPORTED_EXTS:
            stem = os.path.splitext(part)[0]
            new_parts.append(stem + file_suffix + ext)
        elif is_last and "." in part:
            # Unsupported file type — pass through unchanged
            new_parts.append(part)
        else:
            new_parts.append(part + folder_suffix)
    return str(pathlib.PurePosixPath(*new_parts)) if new_parts else rel


def strip_folder_suffix(rel: str, folder_suffix: str) -> str:
    """Reverse mirror: strip folder suffix from each path segment."""
    return "/".join(
        seg[: -len(folder_suffix)] if seg.endswith(folder_suffix) else seg
        for seg in rel.split("/")
    )


def prune_empty_dirs(root: str):
    """
    Walk a directory tree bottom-up and remove any directories that are empty.
    Called after file deletions so orphan folders don't linger on either side.
    """
    for dirpath, dirs, files in os.walk(root, topdown=False):
        if dirpath == root:
            continue
        try:
            if not os.listdir(dirpath):
                os.rmdir(dirpath)
        except OSError:
            pass


def rename_output_suffixes(
    old_folder_suffix: str, new_folder_suffix: str,
    old_file_suffix: str,   new_file_suffix: str,
) -> dict:
    """
    Walk OUTPUT_ROOT bottom-up and rename:
    - folders ending with old_folder_suffix → new_folder_suffix
    - supported image files whose stem ends with old_file_suffix → new_file_suffix
    Returns counts of renamed folders and files.
    """
    renamed_folders = 0
    renamed_files   = 0

    for dirpath, dirs, files in os.walk(OUTPUT_ROOT, topdown=False):
        # Rename matching supported-format files first (while still inside correct dir)
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            stem = os.path.splitext(fname)[0]
            if ext in SUPPORTED_EXTS and old_file_suffix and stem.endswith(old_file_suffix):
                new_stem = stem[: -len(old_file_suffix)] + new_file_suffix
                src = os.path.join(dirpath, fname)
                dst = os.path.join(dirpath, new_stem + ext)
                if src != dst and not os.path.exists(dst):
                    os.rename(src, dst)
                    renamed_files += 1

        # Rename matching folders (bottom-up so children are renamed first)
        folder_name = os.path.basename(dirpath)
        if old_folder_suffix and folder_name.endswith(old_folder_suffix):
            new_name = folder_name[: -len(old_folder_suffix)] + new_folder_suffix
            parent   = os.path.dirname(dirpath)
            src      = dirpath
            dst      = os.path.join(parent, new_name)
            if src != dst and not os.path.exists(dst):
                os.rename(src, dst)
                renamed_folders += 1

    return {"renamed_folders": renamed_folders, "renamed_files": renamed_files}

# ── Filesystem tree ───────────────────────────────────────────────────────────

def build_tree(root: str, rel: str = "") -> dict:
    abs_path = os.path.join(root, rel) if rel else root
    name = os.path.basename(abs_path) or root
    node = {"name": name, "rel": rel, "children": [], "files": []}
    try:
        entries = sorted(os.scandir(abs_path), key=lambda e: (not e.is_dir(), e.name.lower()))
        for entry in entries:
            entry_rel = os.path.join(rel, entry.name).replace("\\", "/") if rel else entry.name
            if entry.is_dir():
                node["children"].append(build_tree(root, entry_rel))
            elif entry.is_file():
                stat = entry.stat()
                ext  = os.path.splitext(entry.name)[1].lower()
                node["files"].append({
                    "name":         entry.name,
                    "rel":          entry_rel,
                    "size":         stat.st_size,
                    "is_supported": ext in SUPPORTED_EXTS,
                })
    except PermissionError:
        pass
    return node


def dir_size(path: str) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def storage_info() -> dict:
    usage = shutil.disk_usage(INPUT_ROOT)
    return {
        "input_used":  dir_size(INPUT_ROOT),
        "output_used": dir_size(OUTPUT_ROOT),
        "disk_total":  usage.total,
        "disk_free":   usage.free,
    }

# ── Model discovery ───────────────────────────────────────────────────────────

def _parse_network_from_folder(folder_name: str) -> str:
    m = re.search(r'--g_network_([A-Za-z0-9]+)', folder_name)
    return m.group(1) if m else ""


# ── Folder overrides ──────────────────────────────────────────────────────────

def load_overrides() -> dict:
    """Returns dict keyed by original_rel → {original_folder, override_folder, filename}."""
    return _load_json(OVERRIDES_FILE, {})

def save_overrides(data: dict):
    _write_json(OVERRIDES_FILE, data, indent=2)


def apply_folder_override(original_rel: str, override_name: str) -> dict:
    """
    Physically rename the folder that contains the model file so the full
    --model_path string encodes the training parameters.
    """
    overrides = load_overrides()
    parts = original_rel.replace("\\", "/").split("/")
    if len(parts) != 2:
        return {"error": "Only single-level model folders are supported "
                         "(expected folder/model.pth)"}

    original_folder, filename = parts
    existing = overrides.get(original_rel)
    current_folder_name = existing["override_folder"] if existing else original_folder

    current_path  = os.path.join(MODEL_ROOT, current_folder_name)
    override_path = os.path.join(MODEL_ROOT, override_name)

    if not os.path.isdir(current_path):
        return {
            "error": f"Folder '{current_folder_name}' not found on disk. "
                     "If it was renamed externally, clear this override first."
        }
    if override_path != current_path and os.path.exists(override_path):
        return {"error": f"A folder named '{override_name}' already exists."}

    if current_path != override_path:
        os.rename(current_path, override_path)

    overrides[original_rel] = {
        "original_folder": original_folder,
        "override_folder": override_name,
        "filename":        filename,
    }
    save_overrides(overrides)
    return {
        "ok":            True,
        "new_rel":       override_name + "/" + filename,
        "new_full_path": os.path.join(MODEL_ROOT, override_name, filename),
    }


def clear_folder_override(original_rel: str) -> dict:
    """
    Rename the folder back to its original name and remove the override entry.
    """
    overrides = load_overrides()
    existing  = overrides.get(original_rel)
    if not existing:
        return {"ok": True, "msg": "No override found for that model"}

    override_path = os.path.join(MODEL_ROOT, existing["override_folder"])
    original_path = os.path.join(MODEL_ROOT, existing["original_folder"])

    if os.path.isdir(override_path) and override_path != original_path:
        if os.path.exists(original_path):
            return {
                "error": f"Cannot revert: a folder named '{existing['original_folder']}' "
                         "already exists. Resolve the conflict manually first."
            }
        os.rename(override_path, original_path)

    del overrides[original_rel]
    save_overrides(overrides)
    return {"ok": True, "original_rel": original_rel}


def _fix_overridden_folders():
    """
    Run before model discovery. For every active override, if the overridden
    folder was externally renamed, locate the model file and rename back.
    As long as an override is active the app owns that folder name.
    """
    overrides = load_overrides()
    for orig_rel, info in overrides.items():
        expected_dir  = os.path.join(MODEL_ROOT, info["override_folder"])
        expected_file = os.path.join(expected_dir, info["filename"])
        if os.path.exists(expected_file):
            continue

        try:
            entries = list(os.scandir(MODEL_ROOT))
        except OSError:
            continue

        for entry in entries:
            if not entry.is_dir():
                continue
            candidate = os.path.join(entry.path, info["filename"])
            if os.path.exists(candidate):
                if not os.path.exists(expected_dir):
                    try:
                        os.rename(entry.path, expected_dir)
                    except OSError:
                        pass
                break


def discover_models() -> list:
    # Heal any externally-renamed overridden folders before scanning
    _fix_overridden_folders()

    overrides = load_overrides()
    # Reverse lookup: (override_folder_name, filename) → original_rel
    override_reverse: dict = {
        (info["override_folder"], info["filename"]): orig_rel
        for orig_rel, info in overrides.items()
    }

    models = []
    host_base = "models/nind_denoise"

    for root, dirs, files in os.walk(MODEL_ROOT):
        for fname in files:
            if fname.endswith((".pt", ".pth")):
                full        = os.path.join(root, fname)
                rel         = os.path.relpath(full, MODEL_ROOT).replace("\\", "/")
                folder_name = os.path.basename(root)
                network     = _parse_network_from_folder(folder_name)

                key          = (folder_name, fname)
                original_rel = override_reverse.get(key, rel)
                ov_info      = overrides.get(original_rel, {})

                models.append({
                    "name":            fname,
                    "full_path":       full,
                    "host_path":       host_base + "/" + rel,
                    "network":         network,
                    "folder":          folder_name,
                    "original_rel":    original_rel,
                    "original_folder": ov_info.get("original_folder", folder_name),
                    "override_folder": ov_info.get("override_folder", ""),
                    "has_override":    bool(ov_info),
                })
    return models

# ── VRAM monitoring ───────────────────────────────────────────────────────────

def get_vram_info() -> list:
    """Returns list of {index, name, used_mb, total_mb, pct} per GPU."""
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        result = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem    = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name   = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            result.append({
                "index":    i,
                "name":     name,
                "used_mb":  round(mem.used  / 1024 / 1024),
                "total_mb": round(mem.total / 1024 / 1024),
                "pct":      round(mem.used / mem.total * 100, 1),
            })
        return result
    except Exception:
        return []


def get_ram_info() -> dict:
    """Returns {used_mb, total_mb, pct} for system RAM.
    Reads /proc/meminfo directly (works in WSL2 and Docker without psutil).
    Falls back to psutil if /proc/meminfo is unavailable."""
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])   # value in kB
        total_kb     = info["MemTotal"]
        available_kb = info.get("MemAvailable", info.get("MemFree", 0))
        used_kb      = total_kb - available_kb
        return {
            "used_mb":  round(used_kb  / 1024),
            "total_mb": round(total_kb / 1024),
            "pct":      round(used_kb / total_kb * 100, 1),
        }
    except Exception:
        pass
    try:
        import psutil
        m = psutil.virtual_memory()
        return {
            "used_mb":  round(m.used  / 1024 / 1024),
            "total_mb": round(m.total / 1024 / 1024),
            "pct":      round(m.percent, 1),
        }
    except Exception:
        return {}


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"

# ── Sync broadcast (set by worker.py init_socketio) ──────────────────────────

_sio = None


def init_sio(sio) -> None:
    """Register the shared SocketIO instance so any route can broadcast."""
    global _sio
    _sio = sio


def broadcast(event: str, data: dict) -> None:
    """Emit a sync event to every connected client on the /sync namespace."""
    if _sio:
        _sio.emit(event, data, namespace="/sync")