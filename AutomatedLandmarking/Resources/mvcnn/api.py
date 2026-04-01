"""
BioFace3D Module 2 – Public Python API for direct integration (e.g. 3D Slicer).

This module exposes a small Python API for model management and landmark prediction.
No CLI, subprocess, or file-based flow is required. Import and call from Python only.

Example:
    from mvcnn.api import predict_landmarks
    landmarks = predict_landmarks("/path/to/face.ply", model_dir="/path/to/models/21Landmarks_25views")
    if landmarks is not None:
        # landmarks is numpy array of shape (N, 3)
        for x, y, z in landmarks:
            ...
"""

from pathlib import Path
import logging
import shutil
import subprocess
import sys
import tempfile
import json
import zipfile

import numpy as np

# Ensure mvcnn is on sys.path so internal imports (e.g. model, map, prediction) resolve
# when the package is used from a different cwd (e.g. Slicer, project root).
_MVCNN_DIR = Path(__file__).resolve().parent
if str(_MVCNN_DIR) not in sys.path:
    sys.path.insert(0, str(_MVCNN_DIR))

_MODEL_CACHE_DIR = Path.home() / ".bioface3d_mvcnn" / "models"
_CONFIGS_DIR = _MVCNN_DIR / "__configs"
_MODEL_METADATA = {
    "21Landmarks_25views": {
        "display_name": "21 landmarks",
        "description": "Recommended full-face landmark model for standard BioFace3D workflows.",
        "landmark_count": 21,
        "recommended": True,
    },
    "20Landmarks_25views": {
        "display_name": "20 landmarks",
        "description": "Depth-only variant with 20 landmarks.",
        "landmark_count": 20,
        "recommended": False,
    },
    "20Landmarks_25v_depth_geom": {
        "display_name": "20 landmarks (depth + geometry)",
        "description": "20-landmark model that uses both geometry and depth renderings.",
        "landmark_count": 20,
        "recommended": False,
    },
    "LYHM_5Landmarks_25views": {
        "display_name": "5 landmarks",
        "description": "Lightweight LYHM model with 5 landmarks.",
        "landmark_count": 5,
        "recommended": False,
    },
}


def _load_config_dict(model_dir):
    config_path = Path(model_dir) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_cached_model_path(model_dir):
    """Return the cache location for model weights."""
    model_dir = Path(model_dir)
    return _MODEL_CACHE_DIR / model_dir.name / "model_best.pth"


def _validate_model_weights(model_path, require_zip=False):
    """
    Return (is_valid, error_message).

    For modern PyTorch checkpoints saved as zip archives, opening the archive is
    enough to catch common corruption cases such as truncated downloads.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        return False, "Model weights file does not exist."
    if model_path.stat().st_size <= 0:
        return False, "Model weights file is empty."

    try:
        with model_path.open("rb") as f:
            signature = f.read(4)
    except OSError as exc:
        return False, f"Could not read model weights: {exc}"

    # Modern torch.save checkpoints are zip archives. Validate the central directory.
    if signature.startswith(b"PK"):
        try:
            with zipfile.ZipFile(str(model_path), "r") as archive:
                names = archive.namelist()
                if not names:
                    return False, "Checkpoint archive is empty."
        except (zipfile.BadZipFile, OSError) as exc:
            return False, f"Checkpoint archive is invalid or truncated: {exc}"
    elif require_zip:
        return False, "Expected a zip-based PyTorch checkpoint."

    return True, None


def get_model_status(model_dir):
    """Return metadata and availability information for a bundled BioFace3D model."""
    model_dir = Path(model_dir)
    config_dict = _load_config_dict(model_dir)
    model_ref = ((config_dict.get("predict") or {}).get("model_pth_or_url") or "").strip()
    bundled_path = model_dir / "model_best.pth"
    cached_path = get_cached_model_path(model_dir)
    candidate_path = Path(model_ref).expanduser() if model_ref else None
    candidate_exists = bool(candidate_path and candidate_path.is_file())
    metadata = dict(_MODEL_METADATA.get(model_dir.name, {}))
    landmark_count = metadata.get("landmark_count")
    if landmark_count is None:
        landmark_count = (((config_dict.get("arch") or {}).get("args") or {}).get("n_landmarks"))
    download_url = model_ref if model_ref.startswith(("http://", "https://")) else None

    resolved_path = None
    availability_source = None
    if bundled_path.is_file():
        resolved_path = bundled_path
        availability_source = "bundled"
    elif cached_path.is_file():
        resolved_path = cached_path
        availability_source = "cached"
    elif candidate_exists:
        resolved_path = candidate_path
        availability_source = "local"

    validation_error = None
    is_available = resolved_path is not None
    if resolved_path is not None:
        is_valid, validation_error = _validate_model_weights(
            resolved_path,
            require_zip=(availability_source == "cached"),
        )
        if not is_valid:
            is_available = False

    return {
        "name": model_dir.name,
        "display_name": metadata.get("display_name", model_dir.name),
        "description": metadata.get("description", ""),
        "landmark_count": landmark_count,
        "recommended": bool(metadata.get("recommended", False)),
        "model_dir": model_dir,
        "config_path": model_dir / "config.json",
        "bundled_weights_path": bundled_path,
        "cached_weights_path": cached_path,
        "resolved_weights_path": resolved_path,
        "is_available": is_available,
        "availability_source": availability_source,
        "download_url": download_url,
        "validation_error": validation_error,
    }


def list_available_models(configs_dir=None):
    """Enumerate bundled BioFace3D model configs with status information."""
    configs_dir = Path(configs_dir) if configs_dir else _CONFIGS_DIR
    models = []
    if not configs_dir.is_dir():
        return models
    for model_dir in sorted(configs_dir.iterdir()):
        if model_dir.is_dir() and (model_dir / "config.json").is_file():
            models.append(get_model_status(model_dir))
    return models


def remove_cached_model(model_dir):
    """Delete cached weights for a model. Bundled weights are never removed."""
    cached_path = get_cached_model_path(model_dir)
    if cached_path.is_file():
        cached_path.unlink()
        return True
    return False


def _ensure_model_weights(model_dir, config_dict):
    """Resolve already-available model weights, or raise if the model has not been downloaded yet."""
    model_path = model_dir / "model_best.pth"
    if model_path.is_file():
        is_valid, validation_error = _validate_model_weights(model_path)
        if not is_valid:
            raise RuntimeError(
                "Bundled model weights for {} are invalid: {}".format(model_dir.name, validation_error)
            )
        return model_path

    cached_path = get_cached_model_path(model_dir)
    if cached_path.is_file():
        is_valid, validation_error = _validate_model_weights(cached_path, require_zip=True)
        if not is_valid:
            raise RuntimeError(
                "Cached model weights for {} are invalid: {}".format(model_dir.name, validation_error)
            )
        return cached_path

    model_ref = (config_dict.get("predict") or {}).get("model_pth_or_url")
    if not model_ref:
        raise FileNotFoundError(
            f"Model not found in {model_dir} and config has no predict.model_pth_or_url entry."
        )

    candidate_path = Path(str(model_ref)).expanduser()
    if candidate_path.is_file():
        is_valid, validation_error = _validate_model_weights(candidate_path)
        if not is_valid:
            raise RuntimeError(
                "Model weights at {} are invalid: {}".format(candidate_path, validation_error)
            )
        return candidate_path

    model_ref_str = str(model_ref)
    if model_ref_str.startswith(("http://", "https://")):
        raise FileNotFoundError(
            "Model weights for {} are not available locally. Download them first with download_model().".format(
                model_dir.name
            )
        )

    raise FileNotFoundError(
        f"Model not found in {model_dir} and configured weights path/URL is unavailable: {model_ref_str}"
    )


def download_model(model_dir, force=False):
    """Download a model into the local cache and return the resolved weights path."""
    model_dir = Path(model_dir)
    status = get_model_status(model_dir)
    if status["is_available"] and not force:
        return status["resolved_weights_path"]
    if status.get("validation_error") and status["availability_source"] in ("bundled", "local"):
        raise RuntimeError(
            "Existing model weights for {} are invalid: {}".format(model_dir.name, status["validation_error"])
        )
    if not status["download_url"]:
        raise FileNotFoundError(f"No download URL configured for {model_dir.name}")
    force = force or (status["availability_source"] == "cached" and bool(status.get("validation_error")))
    return _download_model_weights(
        status["download_url"],
        status["cached_weights_path"],
        force=force,
    )


def _download_model_weights(url, destination, force=False):
    """Download model weights once and reuse the cached file on later runs."""
    if destination.is_file() and destination.stat().st_size > 0 and not force:
        is_valid, validation_error = _validate_model_weights(destination, require_zip=True)
        if is_valid:
            return destination
        force = True
        logging.warning("Discarding invalid cached model weights at %s: %s", destination, validation_error)

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_suffix(destination.suffix + ".part")
    logging.info("Downloading BioFace3D model weights from %s", url)
    try:
        curl_exe = shutil.which("curl.exe") or shutil.which("curl")
        if not curl_exe:
            raise RuntimeError(
                "curl is required to download BioFace3D model weights on this system, but it was not found on PATH."
            )
        _download_with_curl(curl_exe, url, partial_path)
        partial_path.replace(destination)
        is_valid, validation_error = _validate_model_weights(destination, require_zip=True)
        if not is_valid:
            try:
                destination.unlink()
            except OSError:
                pass
            raise RuntimeError(f"Downloaded checkpoint is invalid: {validation_error}")
        return destination
    except Exception as exc:
        if partial_path.exists():
            partial_path.unlink()
        raise RuntimeError(f"Could not download model weights from {url}: {exc}") from exc


def _download_with_curl(curl_exe, url, destination):
    """Download using curl when available; more reliable than urllib on some Windows setups."""
    cmd = [curl_exe, "--fail", "--location", url, "--output", str(destination)]
    kwargs = {"capture_output": True, "text": True}
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.run(cmd, **kwargs)
    if proc.returncode != 0:
        details = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError("curl download failed (exit code {}): {}".format(proc.returncode, details or "unknown error"))
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise RuntimeError("curl download did not produce a valid file.")


def predict_landmarks(
    mesh_path,
    model_dir,
    use_gpu=True,
    predict_num=1,
    max_ransac_error=5.0,
    predict_tries=3,
    output_path=None,
):
    """
    Run landmark prediction on a 3D facial mesh.

    Pure Python API: no subprocess, no CLI. Suitable for use from a Slicer Logic class.

    Parameters
    ----------
    mesh_path : str or pathlib.Path
        Path to the input mesh file (.ply, .obj, .stl, .vtk, .wrl).
    model_dir : str or pathlib.Path
        Directory containing config.json and, optionally, bundled model_best.pth
        (e.g. mvcnn/__configs/21Landmarks_25views). Weights must already be
        available locally, either bundled, cached, or referenced by a local path.
    use_gpu : bool, optional
        Use GPU if available. Default True.
    predict_num : int, optional
        Number of prediction runs to average (default 10; "Mean Predictions" in official UI).
    max_ransac_error : float, optional
        RANSAC error threshold; prediction rejected if ransac_error >= this (default 5.0).
    predict_tries : int, optional
        Max retries per run when ransac exceeds threshold (default 3).
    output_path : str or pathlib.Path or None, optional
        If set, landmark files are written here (e.g. .txt, .json). If None, only the array is returned.

    Returns
    -------
    numpy.ndarray or None
        Landmarks as (N, 3) array of x,y,z, or None if prediction failed.
    """
    mesh_path = Path(mesh_path)
    model_dir = Path(model_dir)

    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Build config dict from JSON and resolve paths
    with config_path.open("r", encoding="utf-8") as f:
        config_dict = json.load(f)

    model_path = _ensure_model_weights(
        model_dir,
        config_dict,
    )
    config_dict.setdefault("predict", {})
    config_dict["predict"]["model_pth_or_url"] = str(model_path.resolve())
    config_dict["n_gpu"] = 1 if use_gpu else 0

    # Create minimal config object for DeepMVLM (no argparse, no CLI)
    config = _minimal_config(
        config_dict,
        use_gpu=use_gpu,
        predict_num=predict_num,
        max_ransac_error=max_ransac_error,
        predict_tries=predict_tries,
        output_path=Path(output_path) if output_path else None,
    )

    # Import here so that the rest of mvcnn is only loaded when the API is used
    from .deepmvlm import DeepMVLM

    dm = DeepMVLM(config)
    basename = mesh_path.stem
    landmarks, _ = dm.predict(
        str(mesh_path.resolve()),
        basename,
        ko_file=None,
        output_path=config.output_path,
    )

    if landmarks is None:
        return None

    return np.asarray(landmarks)


class _MinimalConfig:
    """
    Minimal config object that satisfies DeepMVLM's expectations.
    No argparse, no CLI; built from a config dict and a few overrides.
    """

    def __init__(
        self,
        config_dict,
        use_gpu=True,
        predict_num=10,
        max_ransac_error=5.0,
        predict_tries=3,
        output_path=None,
    ):
        self._config = config_dict
        self._predict_num = predict_num
        self._predict_tries = predict_tries
        self._max_ransac = float(max_ransac_error)
        self._output_format = "json" if output_path else "txt"
        self._output_path = output_path
        self._ngpu = 1 if use_gpu else 0

        tmp = Path(tempfile.gettempdir()) / "bioface3d_mvcnn"
        self._temp_dir = tmp / "temp"
        self._save_dir = tmp / "saved"
        self._log_dir = tmp / "log"
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

    def __getitem__(self, key):
        return self._config[key]

    def initialize(self, name, module, *args, **kwargs):
        """Build model/component from config (same contract as parse_config.ConfigParser)."""
        module_name = self._config[name]["type"]
        module_args = dict(self._config[name]["args"])
        assert all(k not in module_args for k in kwargs), "Overwriting kwargs given in config is not allowed"
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def get_logger(self, name, verbosity=2):
        logger = logging.getLogger(name)
        level = self._log_levels.get(verbosity, logging.WARNING)
        logger.setLevel(level)
        return logger

    @property
    def config(self):
        return self._config

    @property
    def temp_dir(self):
        return self._temp_dir

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def predict_num(self):
        return self._predict_num

    @property
    def predict_tries(self):
        return self._predict_tries

    @property
    def max_ransac(self):
        return self._max_ransac

    @property
    def output_format(self):
        return self._output_format

    @property
    def output_path(self):
        return self._output_path

    @property
    def ngpu(self):
        return self._ngpu


def _minimal_config(
    config_dict,
    use_gpu=True,
    predict_num=10,
    max_ransac_error=5.0,
    predict_tries=3,
    output_path=None,
):
    return _MinimalConfig(
        config_dict,
        use_gpu=use_gpu,
        predict_num=predict_num,
        max_ransac_error=max_ransac_error,
        predict_tries=predict_tries,
        output_path=output_path,
    )
