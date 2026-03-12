# NIND Denoise Manager

A Dockerized web UI that wraps [nind-denoise](https://github.com/trougnouf/nind-denoise) — a deep learning image denoising tool — with a browser-based interface for file management, command configuration, and parallel GPU processing.

> **Note:** nind-denoise has no official Docker setup or management interface. This project provides one.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
  - [① File Manager](#-file-manager)
  - [② Selector & Command Builder](#-selector--command-builder)
  - [③ Parallel Worker Engine](#-parallel-worker-engine)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [File Structure](#file-structure)
- [Typical Workflow](#typical-workflow)
- [Limitations](#limitations)
- [License](#license)
- [A Note on Development](#a-note-on-development)
- [Background & Motivation](#background--motivation)
- [Hardware Requirements](#hardware-requirements)
- [Disclaimer & Upstream Compatibility](#disclaimer--upstream-compatibility)
- [Acknowledgements](#acknowledgements)

---

## Overview


nind-denoise is a powerful denoising tool, but running it directly requires manual command construction, path management, and per-file invocation. This manager adds a GUI layer on top of it, handling:

- Uploading and organizing input images
- Scanning available models and auto-detecting their network architecture
- Generating and previewing denoise commands before running them
- Distributing work across multiple parallel GPU workers
- Real-time progress monitoring and session reporting

Everything runs inside a single Docker container. No separate installation of PyTorch, CUDA toolkits, or Python environments on the host is required beyond Docker and the NVIDIA Container Toolkit.

---

## Features

### ① File Manager


- Upload files and entire folder structures via drag-and-drop or browser picker
- Input and output panes with synchronized folder expansion
- Mirrored output directory structure auto-created on upload (`_nind` folder suffix, `_nindimg` file suffix — both configurable)
- Per-section storage usage display
- Delete, create folders, and download as zip — all locked during active processing to prevent corruption

---

### ② Selector & Command Builder

- File tree with per-file and per-folder checkboxes
- Only supported image formats generate commands — non-supported files are counted and warned about but never block selection
- Model auto-discovery: scans the models directory and parses the `--network` parameter directly from training folder names
- Configurable parameters: network profile, model path, compute device, tile size (cs/ucs), output suffixes
- Advanced optional parameters panel: `--overlap`, `--exif_method`, `--model_parameters`, `--max_subpixels`, `--whole_image`, `--pad`, `--debug`
- Live command preview — queue regenerates automatically on every selection or settings change
- Settings persist across container restarts via bind-mounted state directory

---

### ③ Parallel Worker Engine

- Configurable worker count with round-robin command distribution
- Per-worker cards showing status, progress, elapsed time, current file, and average time per image
- Real-time VRAM and RAM usage bars (per GPU via pynvml)
- Collapsible process log terminal with color-coded output (errors, warnings, success, debug)
- Overwrite or skip existing output — configurable per session
- Session report on completion: total completed, failed, skipped, mean time, throughput
- Graceful cancel mid-session with full report on exit

---

## Requirements

- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Windows + WSL2 or a Linux host (GPU passthrough via WSL2 is fully supported)
- An NVIDIA GPU with a driver that supports CUDA 12.6
  - Driver ≥ 525.60 on Linux
  - Driver ≥ 526.x on Windows / WSL2

See [Hardware Requirements](#hardware-requirements) for VRAM and RAM guidance.

---

## Quick Start

```bash
# 1. Clone this repository
git clone https://github.com/your-username/nind-denoise-manager.git
cd nind-denoise-manager

# 2. Place your downloaded model folders inside:
#    ./nind_denoise_models/

# 3. Build and launch
docker compose up -d --build

# 4. Open in browser
http://localhost:10010
```

Model weights (`.pt` / `.pth` files) are not included. See [Recommended Model Weights](#recommended-model-weights) for download links and placement instructions.

---

## File Structure

```
nind-denoise-manager/
├── dockerfiles/
│   ├── cu126/
│   │   └── Dockerfile           ← Default build (CUDA 12.6, widest GPU support)
│   └── cu128/
│       └── Dockerfile           ← Alternative build (CUDA 12.8, latest GPUs only)
├── docker-compose.yaml
├── app/
│   ├── main.py                  ← Flask + SocketIO entry point
│   ├── routes/
│   │   ├── filemanager.py       ← Upload, delete, download, mkdir
│   │   ├── selector.py          ← Settings, model scan, command generation
│   │   └── worker.py            ← Parallel worker engine (SocketIO)
│   ├── shared/
│   │   └── utils.py             ← Shared helpers, path mirroring, VRAM
│   └── templates/
│       └── index.html           ← Single-page UI (all three tabs)
├── nind_denoise_models/         ← Bind-mounted model weights (you provide these)
│   ├── my_unet_model/           ← Custom folder name of your choice (UNet)
│   │   └── generator_280.pt
│   └── my_utnet_model/          ← Custom folder name of your choice (UtNet)
│       ├── generator_684.pt
│       └── generator_650.pt
├── state/                       ← Persistent state (settings, commands, lock)
└── uploads/
    ├── input/                   ← Input images
    └── output/                  ← Denoised output
```

Only `app/` is baked into the Docker image. Everything else is bind-mounted from the host at runtime.

### Recommended Model Weights

Model weights must be downloaded separately from the [official nind-denoise Google Drive](https://drive.google.com/drive/folders/1XmY9yO3yhhhdwQ_btYCIpkUBFQ88H-pr) and placed inside `./nind_denoise_models/` before starting the container.

Each `.pt` / `.pth` file must be placed inside its own **custom-named subfolder**. The folder name is used by the manager for display and auto-detection — you can name it anything meaningful to you. The official training folder names from the Google Drive are very long strings encoding the full training command, and while they work, they are not required. A short descriptive name is perfectly valid.

The following weights have been personally tested and are recommended as a starting point:

| File | Network |
|---|---|
| `generator_280.pt` | UNet |
| `generator_684.pt` | UtNet |
| `generator_650.pt` | UtNet |

> **Note on folder naming and Windows/Linux compatibility:** The original model folders downloaded from Google Drive use the full training command string as the folder name (e.g. `2019-08-03T16:14_nn_train.py_--g_network_UNet_...`). These names contain characters such as colons (`:`) that are valid on Linux but illegal in Windows folder names. If you are working on a Windows host with WSL2, placing the model files into your own custom-named subfolders avoids this cross-platform naming conflict entirely.

---

## Typical Workflow

1. **File Manager tab** — Upload your input images or folders. The output directory structure is mirrored automatically.
2. **Selector tab** — Select the files to process. Choose a model, verify the network profile is correct, adjust tile size and any advanced parameters, then review the generated command queue.
3. **Worker tab** — Set the number of parallel workers and click **▶ START DENOISE**. Monitor per-worker progress and VRAM usage in real time. Review the session report when complete.

---

## Limitations

This project is designed for **personal or small-scale self-hosted use**. It is not suited for production deployment or multi-user environments. Specific constraints to be aware of:

- **Single user only.** There is no authentication, session isolation, or access control. Do not expose port 10010 to an untrusted network.
- **No training manager.** This tool only handles inference (denoising). Model training is not supported and may be added in a future release.
- **Model loading is per-worker.** Each parallel worker loads the full model into VRAM independently. On a 16 GB card, 2–3 workers is typically the safe ceiling before running out of memory.
- **TIFF and common formats only.** nind-denoise natively processes `.tif`/`.tiff` files. JPEG and PNG support exists in the upstream tool but results may vary. RAW camera formats (CR2, NEF, ARW, etc.) are not supported without external pre-conversion.
- **No persistent session recovery.** If the container is restarted mid-session, the in-progress worker state is lost. The command queue (`commands.json`) persists and can be re-run, but completed/failed status from the interrupted session is not retained.
- **No upload size or rate limiting.** Very large folder uploads may time out depending on browser and network conditions. The server enforces a hard 10 GB per-request upload limit.
- **Cancel is soft.** Cancelling a session sends SIGTERM to the current process group, waits up to 5 seconds, then SIGKILL. The current file may finish or produce partial output depending on when the signal arrives.

---

## License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

It wraps [nind-denoise](https://github.com/trougnouf/nind-denoise), which is also GPL-3.0 licensed. Model weights are subject to their own terms from the original authors.

See [LICENSE](LICENSE) for the full license text.

---

## A Note on Development

This is my first publicly released open-source project. Much of the implementation was developed with the assistance of [Claude](https://claude.ai), an AI assistant made by Anthropic.

I want to be transparent about this because it is an honest reflection of how the project was built. My programming background is practical but narrow — I have working knowledge of basic Python and C++ (primarily in the context of Arduino/embedded systems), and I learned Docker as part of building this project. I do not have a strong foundation in web development, Flask, JavaScript, or SocketIO.

What I do have is a clear understanding of how programming systems work in general, and a specific and well-defined problem I needed to solve. The approach throughout was to provide Claude with detailed, guided prompts — specifying requirements, constraints, edge cases, and workflow logic — and to review, test, and iterate on the output until it matched what I needed. The architecture decisions, the workflow design, and the problem definition were mine. Claude handled the bulk of the implementation in languages and frameworks I would not have been able to write from scratch on my own.

The code works, it has been tested on a real workflow, and the license is open. That is ultimately what matters.

If you are interested in recreating or extending this project using the same AI-assisted approach, a separate guide is provided in [`CLAUDE.md`](CLAUDE.md). It documents the prompting strategy, session setup, and workflow that produced this codebase.

---

## Background & Motivation

This project exists to solve a specific personal problem: denoising high-ISO photography files as part of a local, private workflow.

Most available denoising options come with trade-offs that made them unsuitable for this use case — commercial tools carry subscription costs or perpetual license fees, quality varies significantly between solutions, and cloud-based or online denoisers require uploading raw or high-resolution image files to third-party servers, which raises legitimate privacy concerns for anyone who considers their photography work sensitive or proprietary.

nind-denoise is a fully local, open-source solution that produces competitive quality results. This manager simply makes it practical to use at scale — handling batches of files, managing folder structure, and making the most of available GPU hardware — without requiring manual command-line work for every image.

The self-hosted Docker setup is intentional: the entire pipeline runs on your own machine, your files never leave your system, and the tool can be rebuilt and audited at any time.

This is a niche tool built for a niche workflow. It is shared publicly in case others have the same problem, but it is not designed to be a general-purpose denoising solution or a commercial product.

---

## Hardware Requirements

| Mode | VRAM (minimum) | System RAM (minimum) |
|---|---|---|
| GPU (`--device 0`) | ~3–4 GB per worker | ~1.5 GB per worker |
| CPU (`--device -1`) | None | ~3–4 GB per worker |

| Component | Details |
|---|---|
| GPU | NVIDIA GPU with CUDA 12.6 support (driver ≥ 525.60 Linux / 526.x WSL2) |
| Recommended VRAM | 6 GB or more |

**NVIDIA hardware is strongly recommended.** The denoising workload is a neural network inference pass over tiled image crops — this is exactly the kind of computation GPUs are designed for. A single large TIFF that takes a few seconds on an RTX-class GPU may take several minutes on CPU alone.

**Worker count and VRAM.** Each worker loads the full model into VRAM independently, consuming approximately 3–4 GB of VRAM and 1.5 GB of system RAM per worker in GPU mode. In CPU mode, no VRAM is used but system RAM rises to approximately 3–4 GB per worker instead. The default worker count is **1** — keep it there unless you have verified sufficient headroom on your system. Exceeding VRAM capacity will cause workers to crash or produce failed jobs; the manager has no safeguard against VRAM overload.

It is also worth noting that based on real session reports from this project's own workflow, the time difference between running a single serial worker and running multiple parallel workers is smaller than you might expect. The per-image processing time is the dominant cost, and the overhead of loading multiple model instances into VRAM tends to offset the parallelism gains. For most personal-scale batches, **1 worker is the practical sweet spot**.

If no compatible NVIDIA GPU is available or the NVIDIA Container Toolkit is not configured, the device selector in the Selector tab can be switched to CPU mode. Everything else in the manager works identically — only throughput is affected.

> **Choosing a Dockerfile:** Two CUDA builds are provided. `dockerfiles/cu126/Dockerfile` (default) targets CUDA 12.6 and supports NVIDIA GPUs from Maxwell (GTX 900) through Ada/Hopper. `dockerfiles/cu128/Dockerfile` targets CUDA 12.8 and supports Turing and newer only (RTX 20xx+). If you have a modern GPU and want the latest runtime, switch the `dockerfile:` line in `docker-compose.yaml`.

---

## Disclaimer & Upstream Compatibility

This is an **unofficial** project. It has no affiliation with, endorsement from, or connection to the original nind-denoise authors or repository.

The manager is built against a specific snapshot of the nind-denoise codebase. Because the upstream project is under active development, any of the following changes on their end could silently break or partially break this tool without warning:

- **Script interface changes** — `denoise_image.py` argument names, flags, or defaults being renamed, removed, or restructured will invalidate pre-generated commands. The `--network`, `--model_path`, `--cs`, `--ucs`, `--device`, and `-i`/`-o` arguments are all called directly by this manager.
- **Model folder naming convention changes** — The manager parses `--g_network_<value>` from training folder names to auto-detect the correct `--network` argument. If that naming pattern changes in future releases, auto-detection will silently fall back to the default and may produce incorrect commands.
- **Python or PyTorch API changes** — The Dockerfile pins specific package versions (`torch==2.7.1+cu126`, `numpy==1.26.4`, etc.) to maintain a known-good environment. Upstream dependency changes may require the Dockerfile to be updated accordingly.
- **Repository structure changes** — The Dockerfile clones nind-denoise from GitHub at build time and expects the script at `src/nind_denoise/denoise_image.py`. A repository restructure would require a Dockerfile update before the image can be rebuilt successfully.

**If something stops working after pulling a fresh build**, the first thing to check is whether the upstream nind-denoise repository has changed in a way that affects any of the above. Pinning the git clone to a specific commit hash in the Dockerfile is the most reliable way to freeze behaviour if long-term stability is a priority.

---

## Acknowledgements

This project would not exist without the work of the nind-denoise authors. The denoising quality, the model architectures, the training pipeline, and the pretrained weights are entirely their contribution. This manager is only a convenience layer on top of what they built.

**If you find nind-denoise useful — whether through this manager or directly — please consider supporting the original project:**

- ⭐ Star the upstream repository: [https://github.com/trougnouf/nind-denoise](https://github.com/trougnouf/nind-denoise)
- 📣 Share or cite their work if you use it in any research or publication context
- 🐛 Report issues with the denoising output, model behaviour, or training directly to the nind-denoise repository — those are upstream concerns, not issues with this manager
- 💬 Engage with their project and community

Contributions to this manager (UI improvements, additional file format support, workflow enhancements) are welcome via pull request or issue on this repository, but please set expectations accordingly — responses and merges may be slow or infrequent.

---

## Citation

Please cite Benoit Brummer's original work if you use nind-denoise in any research or publication context:

```bibtex
@InProceedings{Brummer_2019_CVPR_Workshops,
  author    = {Brummer, Benoit and De Vleeschouwer, Christophe},
  title     = {Natural Image Noise Dataset},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month     = {June},
  year      = {2019}
}
```