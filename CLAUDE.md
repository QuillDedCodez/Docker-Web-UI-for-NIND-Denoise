# NIND Denoise Manager — Project Instructions

## What This Project Is
A Dockerized web UI for managing and running [nind-denoise](https://github.com/trougnouf/nind-denoise), a deep learning image denoising tool. The UI is a single-page Flask + SocketIO app with three tabs: File Manager, Selector, and Worker.

---

## Stack
- **Backend:** Python 3.12, Flask 3.0.3, Flask-SocketIO 5.3.6 (`async_mode="threading"`), nvidia-ml-py (imported as `pynvml`)
- **Frontend:** Single HTML file (`index.html`), vanilla JS, dark theme CSS, SocketIO client
- **ML:** PyTorch 2.7.1+cu126, nind-denoise cloned from GitHub (sparse clone — `src/` only)
- **Container:** Docker + docker-compose, single container, port `10010`
- **Host OS:** Windows with WSL2 (GPU passthrough via WSL2 driver)

---

## Folder Structure (host)

```
nind-denoise-manager/
├── dockerfiles/
│   ├── cu126/
│   │   └── Dockerfile           ← Default build target (CUDA 12.6)
│   └── cu128/
│       └── Dockerfile           ← Alternative (CUDA 12.8, newer GPUs only)
├── docker-compose.yaml
├── app/
│   ├── main.py                  ← Flask + SocketIO entry point
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── filemanager.py
│   │   ├── selector.py
│   │   └── worker.py
│   ├── shared/
│   │   ├── __init__.py
│   │   └── utils.py
│   └── templates/
│       └── index.html
├── nind_denoise_models/         ← Bind-mounted model weights
├── state/                       ← Bind-mounted persistent state
└── uploads/
    ├── input/                   ← Bind-mounted to /data/input
    └── output/                  ← Bind-mounted to /data/output
```

---

## Container Paths
| Purpose | Container Path |
|---|---|
| nind-denoise script | `/app/nind-denoise/src/nind_denoise/denoise_image.py` |
| Models root | `/app/nind-denoise/models/nind_denoise` |
| Input data | `/data/input` |
| Output data | `/data/output` |
| State dir (bind mount) | `/app/state` |
| State files | `/app/state/settings.json`, `commands.json`, `lock.json`, `folder_overrides.json`, `selection.json`, `commands_history.json` |

---

## Environment Variables (docker-compose)
```
INPUT_ROOT=/data/input
OUTPUT_ROOT=/data/output
MODEL_ROOT=/app/nind-denoise/models/nind_denoise
STATE_DIR=/app/state
NIND_SCRIPT=/app/nind-denoise/src/nind_denoise/denoise_image.py
FLASK_PORT=10010
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096
```

---

## Docker

### Dockerfile key decisions (cu126)
- Base: `python:3.12-slim`
- `ARG BUILDKIT_INLINE_CACHE=1` — bakes layer cache into image so BuildKit reuses layers after daemon restart or WSL2 reboot
- Layer order maximises cache hits: system deps → PyTorch (cu126) → NumPy → other Python deps → nind-denoise source → app source
- `numpy==1.26.4` is pinned — NumPy 2.x breaks `opencv-python-headless` at import (binary ABI mismatch)
- `opencv-python-headless` not `opencv-python` — no Qt/X11 dependency chain needed in Docker
- `nvidia-ml-py` is the official replacement for the deprecated standalone `pynvml` package; the Python import name is still `pynvml` — no code change required
- Git is installed, used for a **sparse clone** (only `src/` fetched), then purged within the same `RUN` layer so no git binary appears in the final image
- Non-root user `nind` (UID/GID 1000) — matches default Linux desktop and WSL2 user so bind-mounted volumes have correct permissions without extra config
- `security_opt: no-new-privileges:true` and `cap_drop: ALL` in docker-compose

### docker-compose key decisions
- State dir is a **bind mount** (`./state:/app/state`), not a named volume — avoids root-owned volume init, easily inspectable on host without `ls -a`
- Models bind mount: `./nind_denoise_models:/app/nind-denoise/models/nind_denoise` (underscore, no space in container path)
- RAM limits: `memory: 16G` hard limit, `memory: 2G` soft reservation
- GPU: `count: all` — exposes every GPU; change to `count: 1` or `device: ["0"]` to pin a specific GPU

### Build & run
```bash
docker compose up -d --build
```

---

## Runtime Entry Point
- `CMD ["python3", "-m", "app.main"]` with `WORKDIR /app`
- Running as a module (`python3 -m app.main`) is required so relative imports (`from app.routes...`) resolve correctly
- `allow_unsafe_werkzeug=True` is set intentionally — this is a local/self-hosted dev-use container, not a production server
- `MAX_CONTENT_LENGTH = 10 GB` (upload size hard limit in Flask)

---

## Shared State Layer — `utils.py`

All shared state, filesystem helpers, and model discovery live here. Every route imports from `utils.py`. Nothing else should manage these concerns.

### Settings
- `DEFAULT_SETTINGS` defines all keys and their defaults
- `SETTINGS_FILE` = `/app/state/settings.json`
- `load_settings()` / `save_settings()` manage persistence
- **Derived keys** (`locked_networks`, `network_tile_defaults`, `supported_exts`) are **never written to disk** — stripped on every save and re-attached on every load. They are always derived from source code constants, not from whatever the client last sent.
- `_ensure_locked_networks()` guarantees UNet and UtNet are always present in the networks list, even if a client POSTs a settings payload that omits them

### Network tile defaults
```python
_LOCKED_NETWORK_NAMES = ["UNet",  "UtNet"]
_LOCKED_NETWORK_CS    = [440,     504    ]
_LOCKED_NETWORK_UCS   = [320,     480    ]
```
`NETWORK_TILE_DEFAULTS` is the runtime dict built from those arrays. To add a locked network, edit only the three parallel lists — nothing else needs changing.

### Supported extensions
```python
SUPPORTED_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
```
Defined in `DEFAULT_SETTINGS["supported_exts"]` and mirrored to `SUPPORTED_EXTS` set for O(1) lookups.

### `mirror_path(rel, folder_suffix, file_suffix)`
Converts an input-relative path to its mirrored output path:
- Each **folder segment** gets `folder_suffix` appended
- Supported image **file stems** get `file_suffix` inserted before the extension
- Unsupported file types are passed through unchanged

This is the single authoritative implementation of the input→output path mapping. Never reimplement it elsewhere.

### `build_command(input_abs, output_abs, settings)`
Builds the full `python3 denoise_image.py ...` command list and returns a command object `{input, output, cmd, cmdstr, status}`. All optional arguments are data-driven via `_OPTIONAL_SCALAR_ARGS` — only appended when non-empty.

### Lock
```python
set_lock(bool)   # writes lock.json
is_locked()      # reads lock.json
```
All mutating endpoints (upload, delete, mkdir, rename suffixes, clear commands, override) return HTTP 423 when locked.

### Model discovery — `discover_models()`
Walks `MODEL_ROOT`, finds `.pt`/`.pth` files, and for each:
- Parses `--g_network_<Name>` from the folder name via regex → the `--network` argument
- Applies any active folder overrides from `folder_overrides.json`
- Returns `name`, `full_path`, `host_path`, `network`, `folder`, `original_rel`, `has_override`

### Folder overrides — `apply_folder_override()` / `clear_folder_override()`
Allow renaming model folders on disk (e.g. to give a short name to a long training-string folder) and persist the mapping in `folder_overrides.json`. `_fix_overridden_folders()` runs before every model scan to auto-restore any externally-renamed overridden folders.

### VRAM / RAM
```python
get_vram_info()   # list of {index, name, used_mb, total_mb, pct} per GPU via pynvml
get_ram_info()    # {used_mb, total_mb, pct} — reads /proc/meminfo directly, psutil fallback
```

### SocketIO broadcast
```python
init_sio(sio)             # called once by worker.py; registers the shared sio instance
broadcast(event, data)    # emits to all clients on /sync namespace
```

### Selection persistence
```python
load_selection()   # reads selection.json → {files: [], dirs: []}
save_selection()   # writes selection.json
```

---

## Tab: File Manager

- Two panes: INPUT (left) and OUTPUT (right)
- Both panes have independent checkbox selection (`fmFiles`/`fmDirs` for input, `fmoFiles`/`fmoDirs` for output)
- INPUT actions: upload files, upload folders (preserves subfolder structure via `webkitRelativePath`), delete selected, download selected, mkdir
- OUTPUT actions: delete selected, download selected
- Upload: batch progress is broadcast via `/sync` namespace → `upload_batch_start`, `upload_file_done`, `upload_batch_cancel` events so any connected client sees live upload progress
- Delete input files (`section: "both"`) → also deletes mirrored output counterpart via `mirror_path()`
- Delete output files (`section: "output"`) → deletes only the specific output file; does NOT call `mirror_path()` again (the path is already an output path)
- **Empty output folders are kept** — they mirror the input tree structure intentionally; `prune_empty_dirs()` is NOT called after deletes
- Download: two-step async flow — `POST /api/download_prepare` starts a background zip job and returns `job_id`; client polls `GET /api/download_status/<job_id>`; fetches `GET /api/download_result/<job_id>` when ready. Single files skip zipping entirely.
- After every mutating operation, `broadcast("tree_updated", {})` notifies all clients to re-fetch

### Suffix rules
- Input folders → output folders get `folder_suffix` appended (default `_nind`)
- Input supported-format files → output files get `file_suffix` inserted before extension (default `_nindimg`)
- Non-supported files are passed through unchanged
- Changing suffixes in Settings triggers `/api/rename_suffixes` → `rename_output_suffixes()` which walks output bottom-up and renames existing folders/files to match

---

## Tab: Selector

- Left pane: input file tree with checkboxes (`selFiles`/`selDirs`, independent from File Manager sets)
- Right pane: settings + command queue
- Settings: folder suffix, file suffix, device, model selection, network, tile sizes (cs/ucs), advanced params
- **Network:** UNet and UtNet are always present, locked (cannot be removed). Custom networks can be added/removed.
- Only one network is active at a time — the selected one is passed as `--network`
- Model selection auto-applies the `--network` parsed from the folder name
- Folder hint shows full folder name below model path field
- Command queue auto-generates (debounced) whenever selection or settings change
- `saveSettings()` must persist to disk **before** `autoGenerate()` runs — this is critical so `/api/generate` reads the correct model_path and network
- Skip reasons are returned separately: `skip_format` (wrong extension), `skip_missing` (file not on disk), `skip_exists` (overwrite=False and output present)
- Selection is persisted to `selection.json` and broadcast on `/sync`

### Warning indicators
- No model selected → orange warning banner (required, not optional)
- Network mismatch → warning shown, then auto-resolved by applying the parsed network
- Starting Worker with no model → confirmation dialog

---

## Tab: Worker

- Control bar: number of workers (1–16 hard cap), overwrite toggle, Start/Cancel buttons, session pill, connected-clients count
- VRAM bar + RAM bar: live per-GPU memory (pynvml) and system RAM (/proc/meminfo), polled via SocketIO
- Session Report appears **above** the workers grid after session completes
- Workers grid: per-worker cards with status, progress bar, file name, elapsed, avg time
- Process log terminal: hidden by default, toggle via checkbox. Real-time stdout/stderr per worker, color-coded. Auto-scrolls. Clears on new session start.

### Worker success logic
- Success is determined by **output file existence** after the process exits, NOT by `returncode`
- `denoise_image.py` sometimes exits non-zero even when it produces output (warnings to stderr)
- Before running a command when `overwrite=True`, the existing output file is **deleted first** so that post-run existence check is meaningful (a pre-existing file from a previous session would otherwise falsely register as success)
- `returncode` is logged to the terminal for debugging

### Cancel behaviour
- Sets `_session["cancelled"] = True` (readline loops check this flag)
- Immediately sends SIGTERM to every active subprocess's **process group** (covers PyTorch data-loader child processes)
- After 5 s, SIGKILL if still alive
- Stdout pipe is drained and closed to prevent file-descriptor leaks

### Status emit throttle
- Worker card updates from the readline loop are throttled to 0.25 s (max ~4 updates/sec) to avoid polling pynvml excessively
- State-transition updates (file started, file done, session complete) are always emitted immediately (unthrottled)

### Multi-client sync
- When a session starts, `_session["worker_config"]` stores `{num_workers, overwrite}` and is included in every subsequent `worker_update` so remote clients apply the winning config to their UI
- Settings are saved atomically at session start (inside `on_start` before threads launch) — eliminates any race between the old fetch('/api/settings') and socket.emit('start') pattern

### Button states
- Start disabled / Cancel enabled while running
- `showReport()` re-enables Start and disables Cancel on session complete

---

## SocketIO Namespaces

### `/worker` — session control
| Event | Direction | Description |
|---|---|---|
| `connect` | client→server | Pushes current status to connecting client only |
| `start` | client→server | Start session with `{workers, overwrite}` |
| `cancel` | client→server | Cancel session — immediate SIGTERM to all active procs |
| `status` | client→server | Request current status |
| `worker_update` | server→client | Real-time status `{running, workers[], vram[], ram, worker_config}` |
| `terminal_line` | server→client | Single line `{worker_id, text}` |
| `session_complete` | server→client | Session finished with full report |
| `error` | server→client | Error message `{msg}` |

### `/sync` — multi-client real-time sync
| Event | Direction | Description |
|---|---|---|
| `connect` | client→server | Server pushes full current state immediately |
| `disconnect` | client→server | Decrements client count |
| `clients_count` | server→client | `{count}` of connected /sync clients |
| `tree_updated` | server→client | File tree changed — clients re-fetch `/api/tree` |
| `commands_updated` | server→client | `{commands}` — command queue changed |
| `settings_updated` | server→client | Full settings object — settings changed |
| `selection_updated` | server→client | `{files, dirs}` — selection changed |
| `history_updated` | server→client | Session history changed — clients re-fetch |
| `upload_batch_start` | server→client | `{batch_id, title, file_count, files}` |
| `upload_file_done` | server→client | `{batch_id, batch_total, file, file_size, elapsed_ms, saved_count}` |
| `upload_batch_cancel` | server→client | `{batch_id, pending_files}` |
| `upload_file_cancel` | server→client | `{batch_id, file, file_size}` |

---

## API Endpoints

### filemanager.py
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/tree` | Returns `{input, output, storage, locked}` tree JSON |
| GET | `/api/storage` | Disk usage stats with formatted strings |
| POST | `/api/upload` | Upload files; preserves `webkitRelativePath`; broadcasts progress |
| POST | `/api/upload_batch_start` | Register batch start for remote client sync |
| POST | `/api/upload_batch_cancel` | Cancel batch — broadcasts to all clients |
| POST | `/api/upload_file_cancel` | Cancel individual file — broadcasts |
| POST | `/api/upload_cancel_all` | Beacon endpoint (tab close) — cancels all active batches |
| POST | `/api/delete` | Delete file/folder. `section`: `"input"`, `"output"`, or `"both"` |
| POST | `/api/rename_suffixes` | Rename output folders/files when suffix setting changes |
| POST | `/api/mkdir` | Create folder in input (and mirrored output) |
| POST | `/api/download_prepare` | Start async zip job; returns `{job_id, type, total}` |
| GET | `/api/download_result/<job_id>` | Stream completed file or zip |
| GET | `/api/download_status/<job_id>` | Poll `{status, pct, done, total}` |
| POST | `/api/download_cancel/<job_id>` | Cancel in-progress zip |
| POST | `/api/force_unlock` | Emergency: clear stale lock.json after container crash |

### selector.py
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/settings` | Load settings (with derived keys attached) |
| POST | `/api/settings` | Save settings; broadcasts `settings_updated` |
| GET | `/api/models` | Discover .pt/.pth files with network/override metadata |
| POST | `/api/generate` | Generate command queue from `{selected: [...]}` |
| GET | `/api/commands` | Return current command queue |
| POST | `/api/commands/clear` | Clear command queue |
| GET | `/api/selection` | Load persisted selector selection |
| POST | `/api/selection` | Save selector selection; broadcasts `selection_updated` |
| POST | `/api/model_overrides/apply` | Rename a model folder on disk; persist override |
| POST | `/api/model_overrides/clear` | Revert a model folder rename; remove override |

### worker.py (SocketIO `/worker` + REST fallbacks)
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/worker/status` | Current session status (REST fallback) |
| GET | `/api/worker/report` | Current session report |
| GET | `/api/commands/history` | Last completed session's commands |
| DELETE | `/api/commands/history` | Clear history file |

---

## Known Gotchas

1. **Module import path** — must run as `python3 -m app.main` from `WORKDIR /app`. Running the file directly breaks relative imports.

2. **Output delete path** — when `section="output"`, `rel_path` is already an output path (with suffixes applied). Do NOT run it through `mirror_path()` again — it will double-suffix and miss the target.

3. **Derived keys are never persisted** — `locked_networks`, `network_tile_defaults`, and `supported_exts` are stripped from `settings.json` on every save and re-built from source on every load. A client POSTing these keys back will have them silently dropped.

4. **Selection isolation** — File Manager input, File Manager output, and Selector each have their own independent checkbox selection sets. Checking files in one tab never affects another.

5. **Worker success = file existence** — not `returncode`. `denoise_image.py` exits non-zero on warnings. Always check `os.path.exists(out_path)`, never trust `proc.returncode` alone.

6. **Overwrite pre-delete** — when `overwrite=True`, the existing output file is removed before launching the subprocess. This makes the post-run existence check meaningful. Without this, a pre-existing file from a prior session would falsely register as completed.

7. **Model root path** — runtime path via docker-compose env is `/app/nind-denoise/models/nind_denoise` (underscore). The fallback default in `utils.py` uses a space (`nind denoise`) — this is a legacy default that is never reached when running via docker-compose. Always use the env var path.

8. **pynvml import name** — the package is installed as `nvidia-ml-py` but imported as `pynvml`. Do not change the import.

9. **State dir bind mount** — uses `./state` (no leading dot) so it's visible without `ls -a` for easy debug inspection of `settings.json`, `lock.json`, etc.

10. **`allow_unsafe_werkzeug=True`** — set intentionally in `sio.run()`. This is a self-hosted local container; the warning can be safely ignored.

11. **Empty output folders** — output folders that become empty after a file delete are **kept**, not pruned. They mirror the input tree structure and should persist to indicate where outputs are expected.

12. **Multi-client race on Start** — if two clients click Start simultaneously, only one wins. The loser immediately receives a `worker_update` with the winning config so it syncs its UI. `worker_config` is cleared when the session ends.