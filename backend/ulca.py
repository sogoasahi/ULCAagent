from typing import Any, Dict, List, Optional, Union
import numpy as np
from scipy import linalg

try:
    from factor_analyzer import Rotator
except ImportError:
    Rotator = None


class EVDULCA:
    """EVDULCA: User-guided Linear Component Analysis via Eigenvalue Decomposition."""

    def __init__(
        self,
        n_components: int = 2,
        apply_varimax: bool = False,
        apply_consist_axes: bool = True,
        verbosity: int = 0,
    ):
        self.n_components = n_components
        self.apply_varimax = apply_varimax
        self.apply_consist_axes = apply_consist_axes
        self.verbosity = verbosity
        self.M: Optional[np.ndarray] = None
        self.alpha: Optional[float] = None
        self.max_alpha: Optional[float] = None
        self.Covs: Dict[str, Any] = {}
        self.gamma0: float = 0.0
        self.gamma1: float = 0.0

    def _calc_covariances(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[int, Dict[str, np.ndarray]]:
        labels = np.unique(y)
        X_c = X - X.mean(axis=0)
        class_mean = {label: X_c[y == label, :].mean(axis=0) for label in labels}
        X_class_mean = np.zeros(X.shape)
        for label in labels:
            X_class_mean[y == label, :] = class_mean[label]

        _Covs = {}
        for label in labels:
            WI = X_c[y == label, :] - X_class_mean[y == label, :]
            BW = X_class_mean[y == label, :]
            Cov_within = (WI.T @ WI) / len(X_c)
            Cov_between = (BW.T @ BW) / len(X_c)
            _Covs[label] = {"within": Cov_within, "between": Cov_between}
        return _Covs

    def _apply_evd(self, C0: np.ndarray, C1: np.ndarray, alpha: float) -> np.ndarray:
        C = C0 - alpha * C1
        if C.shape[0] > 64:
            _, v = linalg.eigh(C, subset_by_index=[C.shape[0]-self.n_components, C.shape[0]-1])
            return v[:, ::-1]
        w, v = linalg.eig(C)
        idx = np.argsort(-np.real(w))[: self.n_components]
        return np.real(v[:, idx])

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w_tg: Union[Dict[int, float], List[float]],
        w_bg: Union[Dict[int, float], List[float]],
        w_bw: Union[Dict[int, float], List[float]],
        alpha: Optional[float] = None,
        gamma0: Optional[float] = None,
        gamma1: Optional[float] = None,
        convergence_ratio: float = 1e-2,
        max_iter: int = 100,
    ) -> "EVDULCA":
        if isinstance(w_tg, list):
            w_tg = {i: w for i, w in enumerate(w_tg)}
        if isinstance(w_bg, list):
            w_bg = {i: w for i, w in enumerate(w_bg)}
        if isinstance(w_bw, list):
            w_bw = {i: w for i, w in enumerate(w_bw)}

        n_feats = X.shape[1]
        std = np.std(X, axis=0)
        active_mask = std > 1e-7
        if not np.any(active_mask):
            self.M = np.zeros((n_feats, self.n_components))
            self.alpha = 0.0
            return self

        X_act = X[:, active_mask]
        d_act = X_act.shape[1]
        labels = np.unique(y)

        Covs = self._calc_covariances(X_act, y)
        self.Covs = Covs

        C0 = np.zeros((d_act, d_act))
        C1 = np.zeros((d_act, d_act))
        w_tg_total = sum(w_tg.get(k, 0.0) for k in labels)
        w_bg_total = sum(w_bg.get(k, 0.0) for k in labels)
        w_bw_total = sum(w_bw.get(k, 0.0) for k in labels)

        for k in labels:
            C0 += w_tg.get(k, 0.0) * Covs[k]["within"] + w_bw.get(k, 0.0) * Covs[k]["between"]
            C1 += w_bg.get(k, 0.0) * Covs[k]["within"]

        if gamma0 is None:
            gamma0 = 1.0 if (w_tg_total + w_bw_total == 0) else 1e-4
        if gamma1 is None:
            gamma1 = 1.0 if (w_bg_total == 0) else 1e-3
        self.gamma0 = gamma0
        self.gamma1 = gamma1

        tr_C0 = np.trace(C0) / d_act if np.trace(C0) > 0 else 1.0
        tr_C1 = np.trace(C1) / d_act if np.trace(C1) > 0 else 1.0
        C0 += gamma0 * tr_C0 * np.eye(d_act)
        C1 += gamma1 * tr_C1 * np.eye(d_act)

        if alpha is not None:
            self.alpha = alpha
            M_act = self._apply_evd(C0, C1, alpha)
        else:
            self.alpha = 0.0
            M_act = self._apply_evd(C0, C1, self.alpha)
            for _ in range(max_iter):
                denom = np.trace(M_act.T @ C1 @ M_act)
                if denom <= 1e-12:
                    break
                prev_alpha = self.alpha
                self.alpha = float(np.trace(M_act.T @ C0 @ M_act) / denom)
                M_act = self._apply_evd(C0, C1, self.alpha)
                if abs(prev_alpha - self.alpha) / (self.alpha + 1e-8) < convergence_ratio:
                    break

        if self.apply_varimax and self.n_components > 1 and Rotator is not None:
            try:
                M_act = Rotator(method="varimax").fit_transform(M_act)
            except Exception:
                pass

        if self.apply_consist_axes:
            col_signs = np.sign(M_act.sum(axis=0))
            col_signs[col_signs == 0] = 1
            M_act = M_act * col_signs
            M_act = M_act[:, np.argsort(-M_act.max(axis=0))]

        self.M = np.zeros((n_feats, self.n_components))
        self.M[active_mask, :] = M_act
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.M is None:
            raise ValueError("Model is not fitted yet.")
        return X @ self.M

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w_tg: Union[Dict[int, float], List[float]],
        w_bg: Union[Dict[int, float], List[float]],
        w_bw: Union[Dict[int, float], List[float]],
        **kwargs,
    ) -> np.ndarray:
        self.fit(X, y, w_tg, w_bg, w_bw, **kwargs)
        return self.transform(X)
