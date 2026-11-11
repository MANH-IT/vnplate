from __future__ import annotations

import numpy as np


class KalmanBoxFilter:
    """Constant-velocity Kalman filter over boxes in (cx, cy, w, h) form."""

    def __init__(self, bbox_cxcywh: np.ndarray):
        self.ndim = 8
        self.dt = 1.0
        self.F = np.eye(self.ndim)
        for i in range(4):
            self.F[i, i + 4] = self.dt
        self.H = np.eye(4, self.ndim)
        self.R = np.eye(4) * 10.0
        self.Q = np.eye(self.ndim)
        self.Q[4:, 4:] *= 0.01
        self.P = np.eye(self.ndim) * 10.0
        self.P[4:, 4:] *= 1000.0
        self.x = np.zeros(self.ndim)
        self.x[:4] = bbox_cxcywh

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4].copy()

    def update(self, bbox_cxcywh: np.ndarray) -> np.ndarray:
        z = np.asarray(bbox_cxcywh, dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.ndim) - K @ self.H) @ self.P
        return self.x[:4].copy()

    def state(self) -> np.ndarray:
        return self.x[:4].copy()

    def velocity(self) -> np.ndarray:
        return self.x[4:].copy()
