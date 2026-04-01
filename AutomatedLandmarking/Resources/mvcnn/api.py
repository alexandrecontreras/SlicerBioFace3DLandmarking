"""
BioFace3D Module 2 – Public Python API for direct integration (e.g. 3D Slicer).

This module exposes a single function-based API. No CLI, subprocess, or file-based
flow is required. Import and call from Python only.

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
import sys
import tempfile
import json
import urllib.request

import numpy as np

# Ensure mvcnn is on sys.path so internal imports (e.g. model, map, prediction) resolve
# when the package is used from a different cwd (e.g. Slicer, project root).
_MVCNN_DIR = Path(__file__).resolve().parent
if str(_MVCNN_DIR) not in sys.path:
    sys.path.insert(0, str(_MVCNN_DIR))

_MODEL_CACHE_DIR = Path.home() / ".bioface3d_mvcnn" / "models"


def _ensure_model_weights(model_dir, config_dict):
    """Resolve bundled weights or download them on demand from the configured URL."""
    model_path = model_dir / "model_best.pth"
    if model_path.is_file():
        return model_path

    model_ref = (config_dict.get("predict") or {}).get("model_pth_or_url")
    if not model_ref:
        raise FileNotFoundError(
            f"Model not found in {model_dir} and config has no predict.model_pth_or_url entry."
        )

    candidate_path = Path(str(model_ref)).expanduser()
    if candidate_path.is_file():
        return candidate_path

    model_ref_str = str(model_ref)
    if model_ref_str.startswith(("http://", "https://")):
        cache_path = _MODEL_CACHE_DIR / model_dir.name / "model_best.pth"
        return _download_model_weights(model_ref_str, cache_path)

    raise FileNotFoundError(
        f"Model not found in {model_dir} and configured weights path/URL is unavailable: {model_ref_str}"
    )


def _download_model_weights(url, destination):
    """Download model weights once and reuse the cached file on later runs."""
    if destination.is_file() and destination.stat().st_size > 0:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_suffix(destination.suffix + ".part")
    logging.info("Downloading BioFace3D model weights from %s", url)
    try:
        with urllib.request.urlopen(url) as response, partial_path.open("wb") as out:
            shutil.copyfileobj(response, out)
        partial_path.replace(destination)
        return destination
    except Exception as exc:
        if partial_path.exists():
            partial_path.unlink()
        raise RuntimeError(f"Could not download model weights from {url}: {exc}") from exc


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
        (e.g. mvcnn/__configs/21Landmarks_25views). If weights are not bundled,
        they are downloaded from predict.model_pth_or_url and cached locally.
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
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    model_path = _ensure_model_weights(model_dir, config_dict)
    config_dict["predict"] = {"model_pth_or_url": str(model_path.resolve())}
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
