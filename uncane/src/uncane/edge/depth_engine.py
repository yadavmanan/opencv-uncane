"""Monocular depth estimation on the edge with OpenCV 5 DNN (ONNX Runtime optional).

Supported models (see scripts/download_models.py):
  * MiDaS v2.1 small           - relative inverse depth, fast, the default
  * Depth-Anything-V2 small    - relative inverse depth, sharper, slower
  * Depth-Anything-V2 metric   - depth in meters (set depth.output = metric)
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_OPENCV_TARGETS = {
    "cpu": (cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU),
    "opencl": (cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_OPENCL),
    "cuda": (cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA),
    "cuda_fp16": (cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA_FP16),
}


def _read_onnx(path: str, cpu: bool = True) -> "cv2.dnn.Net":
    # OpenCV 5's new graph engine handles dynamic input sizes (the classic engine
    # fails on MiDaS reshapes at sizes other than the export size) but only runs
    # on CPU. GPU targets (CUDA/OpenCL) need the classic engine.
    engine = getattr(cv2.dnn, "ENGINE_NEW" if cpu else "ENGINE_CLASSIC", None)
    if engine is not None:
        try:
            return cv2.dnn.readNetFromONNX(path, engine=engine)
        except (cv2.error, TypeError) as exc:
            log.warning("ENGINE_NEW failed (%s); falling back to default engine", exc)
    return cv2.dnn.readNetFromONNX(path)


class DepthEngine:
    """Runs a depth network and returns a float32 map at network resolution.

    For ``output == "relative_inverse"`` larger values mean closer (disparity-like,
    arbitrary scale/shift). For ``output == "metric"`` values are meters.
    """

    def __init__(
        self,
        model_path: str,
        input_size: int = 160,
        output: str = "relative_inverse",
        backend: str = "opencv",
        opencv_target: str = "cpu",
        input_name: Optional[str] = None,
    ):
        if output not in ("relative_inverse", "metric"):
            raise ValueError("depth.output must be relative_inverse or metric")
        self.output = output
        self.input_size = int(input_size)
        self.backend = backend
        if backend == "opencv":
            cpu = opencv_target not in _OPENCV_TARGETS or opencv_target == "cpu"
            self.net = _read_onnx(model_path, cpu=cpu)
            if not cpu:
                b, t = _OPENCV_TARGETS[opencv_target]
                self.net.setPreferableBackend(b)
                self.net.setPreferableTarget(t)
        elif backend == "onnxruntime":
            import onnxruntime as ort  # optional dependency

            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                         if p in ort.get_available_providers()]
            self.session = ort.InferenceSession(model_path, providers=providers)
            inp = self.session.get_inputs()[0]
            self.input_name = input_name or inp.name
            fixed = inp.shape[-1]
            if isinstance(fixed, int) and fixed != self.input_size:
                log.info("model has fixed input %d; overriding depth.input_size", fixed)
                self.input_size = fixed
        else:
            raise ValueError(f"unknown depth backend {backend!r}")
        log.info("depth engine: %s backend, %dpx, %s output", backend, self.input_size, output)

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        s = self.input_size
        img = cv2.resize(frame_bgr, (s, s), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        return np.ascontiguousarray(img.transpose(2, 0, 1)[None])

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        blob = self.preprocess(frame_bgr)
        if self.backend == "opencv":
            self.net.setInput(blob)
            out = self.net.forward()
        else:
            out = self.session.run(None, {self.input_name: blob})[0]
        depth = np.squeeze(out).astype(np.float32)
        if depth.ndim != 2:
            raise RuntimeError(f"unexpected depth output shape {out.shape}")
        return depth


def colorize(depth: np.ndarray, metric: bool, size: Optional[tuple] = None) -> np.ndarray:
    """Debug visualisation: near = warm colours."""
    if metric:
        vis = np.clip(1.0 - depth / 5.0, 0.0, 1.0)
    else:
        lo, hi = np.percentile(depth, (2, 98))
        vis = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    img = cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST) if size else img
