"""
dataset_generator.py
=====================
Generates training data for the ADAPTIVE MODULATION SELECTION classifier.

IMPORTANT DESIGN RULE (this is the fix for the target-leakage bug):
    - BER_BPSK, BER_QPSK, BER_16QAM are computed and SAVED for inspection
      and plotting only.
    - They are NEVER used as inputs to the neural network.
    - Best_Modulation (the label) is derived FROM the BER values using a
      fixed rule, but once that label is written, the BER values that
      created it are discarded from the model's point of view.
    - The classifier in classifier.py only ever sees SNR_dB as input.

This mirrors the real system: a transmitter cannot measure BER directly
without decoding future data first. It CAN estimate SNR (via pilot
symbols, channel sounding, etc.) and must decide a modulation scheme
based on that estimate alone. That is the problem this project solves.

Run:
    python dataset_generator.py
Requires BPSK.py, QPSK.py, QAM16.py (with Gray-coded 16-QAM and the
physically-measured noise formula) in the same folder.
"""

import numpy as np
import pandas as pd

from BPSK  import bpsk_modulate,  bpsk_demodulate,  compute_ber, add_awgn_noise as awgn_bpsk
from QPSK  import qpsk_modulate,  qpsk_demodulate,                add_awgn_noise as awgn_qpsk
from QAM16 import qam16_modulate, qam16_demodulate,               add_awgn_noise as awgn_qam16


# ── Reproducibility ─────────────────────────────────────────────────────
rng = np.random.default_rng(42)
np.random.seed(42)   # add_awgn_noise() in each module uses np.random directly


# ── Labelling rule ───────────────────────────────────────────────────────
BER_THRESHOLD = 1e-2   # acceptable BER limit for "this scheme works here"


def get_best_modulation(ber_bpsk, ber_qpsk, ber_qam16):
    """
    Priority: use the most spectrally efficient scheme that still meets
    the BER requirement. Fall back to the most robust scheme (BPSK) if
    nothing meets the threshold.
    """
    if ber_qam16 < BER_THRESHOLD:
        return "16-QAM"
    elif ber_qpsk < BER_THRESHOLD:
        return "QPSK"
    else:
        return "BPSK"


# ── One simulation trial at a given SNR ────────────────────────────────
def simulate_one_trial(snr_db, N):
    bits_bpsk  = rng.integers(0, 2, N)
    bits_qpsk  = rng.integers(0, 2, N)
    bits_qam16 = rng.integers(0, 2, N)

    ber_bpsk  = compute_ber(bits_bpsk,  bpsk_demodulate(awgn_bpsk(bpsk_modulate(bits_bpsk), snr_db)))
    ber_qpsk  = compute_ber(bits_qpsk,  qpsk_demodulate(awgn_qpsk(qpsk_modulate(bits_qpsk), snr_db)))
    ber_qam16 = compute_ber(bits_qam16, qam16_demodulate(awgn_qam16(qam16_modulate(bits_qam16), snr_db)))

    return ber_bpsk, ber_qpsk, ber_qam16


# ── Full dataset generation ────────────────────────────────────────────
def generate_dataset(snr_range, trials_per_snr, N_per_trial):
    n_snrs = len(snr_range)
    total_rows = n_snrs * trials_per_snr

    snr_arr       = np.empty(total_rows)
    trial_arr     = np.empty(total_rows, dtype=int)
    ber_bpsk_arr  = np.empty(total_rows)
    ber_qpsk_arr  = np.empty(total_rows)
    ber_qam16_arr = np.empty(total_rows)
    best_mod_arr  = np.empty(total_rows, dtype=object)

    idx = 0
    for i, snr_db in enumerate(snr_range):
        if i % 10 == 0:
            print(f"  Progress: SNR = {snr_db:.1f} dB ({i+1}/{n_snrs})")

        for trial in range(trials_per_snr):
            ber_bpsk, ber_qpsk, ber_qam16 = simulate_one_trial(snr_db, N_per_trial)
            best_mod = get_best_modulation(ber_bpsk, ber_qpsk, ber_qam16)

            snr_arr[idx]       = snr_db
            trial_arr[idx]     = trial + 1
            ber_bpsk_arr[idx]  = ber_bpsk
            ber_qpsk_arr[idx]  = ber_qpsk
            ber_qam16_arr[idx] = ber_qam16
            best_mod_arr[idx]  = best_mod
            idx += 1

    df = pd.DataFrame({
        "SNR_dB":          snr_arr,
        "Trial":           trial_arr,
        "BER_BPSK":        ber_bpsk_arr,   # kept for inspection/plots only
        "BER_QPSK":        ber_qpsk_arr,   # NOT fed to the classifier
        "BER_16QAM":       ber_qam16_arr,  # NOT fed to the classifier
        "Best_Modulation": best_mod_arr,   # the label
    })
    return df


if __name__ == "__main__":
    snr_range      = np.arange(-20, 21, step=0.5)   # widen to -20..30 later if you want
    trials_per_snr = 10
    N_per_trial    = 100_000

    print("=" * 60)
    print("  Dataset Generation — Adaptive Modulation Selection")
    print("=" * 60)
    print(f"  SNR range      : {snr_range[0]} dB to {snr_range[-1]} dB")
    print(f"  Trials per SNR : {trials_per_snr}")
    print(f"  Bits per trial : {N_per_trial:,}")
    print("-" * 60)

    df = generate_dataset(snr_range, trials_per_snr, N_per_trial)
    df.to_csv("modulation_dataset.csv", index=False)

    print("-" * 60)
    print(f"Dataset saved: {len(df)} rows -> modulation_dataset.csv")
    print("\nFirst few rows:")
    print(df.head(10).to_string(index=False))

    print("\nLabel distribution:")
    print(df["Best_Modulation"].value_counts())

    print("\nSNR -> Best Modulation (first trial per SNR):")
    print(df.groupby("SNR_dB")["Best_Modulation"].first().to_string())