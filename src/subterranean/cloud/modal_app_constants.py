"""Modal app primitives shared between :mod:`modal_app` and :mod:`_modal_serve`.

Lives in its own module so :mod:`_modal_serve` (which must avoid
``from __future__ import annotations`` for Modal's parameter introspection)
can share the same :class:`modal.App`, images, volumes, and serve-knob
constants as the rest of the cloud package. Importing this module requires
``modal``.
"""

from __future__ import annotations

import modal

APP_NAME = "subterranean"
SERVE_APP = modal.App(APP_NAME)

# Each image pulls only the extras the step needs, keeping cold-starts lean.
_PIP_PKG = "subterranean-agents"

#: CPU image for the API-bound generate/evaluate steps (core + anthropic only).
CPU_IMAGE = modal.Image.debian_slim(python_version="3.11").pip_install(f"{_PIP_PKG}[report]")

#: Training image: the heavy ML stack (torch/trl/deepspeed/bitsandbytes).
TRAIN_IMAGE = modal.Image.debian_slim(python_version="3.11").pip_install(f"{_PIP_PKG}[train]")

#: Serving image: vLLM (CUDA/Linux only).
SERVE_IMAGE = modal.Image.debian_slim(python_version="3.11").pip_install(f"{_PIP_PKG}[serve]")

#: Persisted build artifacts (flowchart IR, dataset.jsonl, eval reports).
BUILD_VOLUME = modal.Volume.from_name("subterranean-build", create_if_missing=True)
#: Persisted fine-tuned model weights.
MODEL_VOLUME = modal.Volume.from_name("subterranean-models", create_if_missing=True)

BUILD_ROOT = "/build"
MODEL_ROOT = "/models"
VOLUMES = {BUILD_ROOT: BUILD_VOLUME, MODEL_ROOT: MODEL_VOLUME}

#: Anthropic API key, injected into the API-bound functions.
ANTHROPIC_SECRET = modal.Secret.from_name("anthropic-secret")

# Timeouts (seconds). Generation/eval are long API-bound jobs; the 3B run is the
# paper's ~3.5h; the 8B ZeRO-3 run is fast (~15-30 min) but gets head-room.
HOUR = 60 * 60
GENERATE_TIMEOUT = 6 * HOUR
TRAIN_3B_TIMEOUT = 6 * HOUR
TRAIN_8B_TIMEOUT = 3 * HOUR
EVALUATE_TIMEOUT = 4 * HOUR

# Serve knobs.
SERVE_GPU = "A100-80GB"
SERVE_TIMEOUT = HOUR
SERVE_SCALEDOWN_WINDOW = 300
SERVE_MIN_CONTAINERS = 0
SERVE_MAX_CONTAINERS = 4
SERVE_PORT = 8000
