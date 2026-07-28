"""Feature extraction utilities for digital communications signals."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

FEATURE_COLUMNS: list[str] = [
	"length",
	"mean_real",
	"mean_imag",
	"std_real",
	"std_imag",
	"std_mag",
	"var_real",
	"var_imag",
	"rms",
	"peak",
	"crest_factor",
	"energy",
	"zero_crossing_rate",
	"iq_correlation",
	"magnitude_entropy",
]


def _safe_entropy(values: np.ndarray) -> float:
	if values.size == 0:
		return 0.0
	total = float(np.sum(values))
	if total <= 0:
		return 0.0
	p = values / total
	p = p[p > 0]
	return float(-np.sum(p * np.log2(p)))


def extract_features(signal: Sequence[complex] | np.ndarray) -> Dict[str, Any]:
	"""Extract basic statistical features from a 1-D signal."""

	x = np.asarray(signal)
	if x.ndim != 1:
		x = np.ravel(x)

	if x.size == 0:
		return {
			"length": 0,
			"mean_real": 0.0,
			"mean_imag": 0.0,
			"std_real": 0.0,
			"std_imag": 0.0,
			"std_mag": 0.0,
			"var_real": 0.0,
			"var_imag": 0.0,
			"rms": 0.0,
			"peak": 0.0,
			"crest_factor": 0.0,
			"energy": 0.0,
			"zero_crossing_rate": 0.0,
			"iq_correlation": 0.0,
			"magnitude_entropy": 0.0,
		}

	xr = np.real(x).astype(float)
	xi = np.imag(x).astype(float)
	mag = np.abs(x).astype(float)

	rms = float(np.sqrt(np.mean(np.square(mag))))
	peak = float(np.max(mag))
	energy = float(np.sum(np.square(mag)))
	crest_factor = float(peak / rms) if rms > 0 else 0.0

	if xr.size > 1:
		zero_crossings = np.sum(np.diff(np.signbit(xr)) != 0)
		zcr = float(zero_crossings / (xr.size - 1))
	else:
		zcr = 0.0

	if np.std(xr) > 0 and np.std(xi) > 0:
		iq_corr = float(np.corrcoef(xr, xi)[0, 1])
	else:
		iq_corr = 0.0

	hist, _ = np.histogram(mag, bins=min(32, max(1, mag.size // 2)), density=False)
	magnitude_entropy = _safe_entropy(hist.astype(float))

	return {
		"length": int(x.size),
		"mean_real": float(np.mean(xr)),
		"mean_imag": float(np.mean(xi)),
		"std_real": float(np.std(xr)),
		"std_imag": float(np.std(xi)),
		"std_mag": float(np.std(mag)),
		"var_real": float(np.var(xr)),
		"var_imag": float(np.var(xi)),
		"rms": rms,
		"peak": peak,
		"crest_factor": crest_factor,
		"energy": energy,
		"zero_crossing_rate": zcr,
		"iq_correlation": iq_corr,
		"magnitude_entropy": float(magnitude_entropy),
	}
