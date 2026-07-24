import numpy as np  # type: ignore[import-not-found]
import pandas as pd  # type: ignore[import-not-found]

from BPSK  import bpsk_modulate, bpsk_demodulate, compute_ber
from QPSK  import qpsk_modulate, qpsk_demodulate
from QAM16 import qam16_modulate, qam16_demodulate

from BPSK  import add_awgn_noise as awgn_bpsk
from QPSK  import add_awgn_noise as awgn_qpsk
from QAM16 import add_awgn_noise as awgn_qam16

BER_THRESHOLD = 1e-2
TRIALS = 10
BITS_PER_TRIAL = 100_000

rng = np.random.default_rng(42)

def get_best_modulation(ber_bpsk, ber_qpsk, ber_qam16):
    if ber_qam16 < BER_THRESHOLD:
        return "16-QAM"
    elif ber_qpsk < BER_THRESHOLD:
        return "QPSK"
    elif ber_bpsk < BER_THRESHOLD:
        return "BPSK"
    else:
        return "BPSK"

def simulate(modulate, demodulate, awgn, bits, snr):
    symbols = modulate(bits)
    received = awgn(symbols, snr)
    bits_hat = demodulate(received)
    return compute_ber(bits, bits_hat)

def generate_dataset(snr_range, trials_per_snr, N_per_trial):
    n_snrs = len(snr_range)
    total_rows = n_snrs * trials_per_snr

    # Preallocate BEFORE the loop, not inside it
    ber_bpsk_arr  = np.empty(total_rows)
    ber_qpsk_arr  = np.empty(total_rows)
    ber_qam16_arr = np.empty(total_rows)
    snr_arr       = np.empty(total_rows)
    trial_arr     = np.empty(total_rows, dtype=int)
    best_mod_arr  = np.empty(total_rows, dtype=object)

    idx = 0
    for i, snr_db in enumerate(snr_range):
        # Progress indicator INSIDE the function, INSIDE the snr loop
        if i % 10 == 0:
            print(f"  Progress: SNR = {snr_db:.1f} dB ({i+1}/{len(snr_range)})")

        for trial in range(trials_per_snr):
            bits      = rng.integers(0, 2, N_per_trial)
            ber_bpsk  = simulate(bpsk_modulate,  bpsk_demodulate,  awgn_bpsk,  bits, snr_db)
            ber_qpsk  = simulate(qpsk_modulate,  qpsk_demodulate,  awgn_qpsk,  bits, snr_db)
            ber_qam16 = simulate(qam16_modulate, qam16_demodulate, awgn_qam16, bits, snr_db)
            best_mod  = get_best_modulation(ber_bpsk, ber_qpsk, ber_qam16)

            ber_bpsk_arr[idx]  = ber_bpsk
            ber_qpsk_arr[idx]  = ber_qpsk
            ber_qam16_arr[idx] = ber_qam16
            snr_arr[idx]       = snr_db
            trial_arr[idx]     = trial + 1
            best_mod_arr[idx]  = best_mod
            idx += 1

    df = pd.DataFrame({
        "SNR_dB":          snr_arr,
        "Trial":           trial_arr,
        "BER_BPSK":        ber_bpsk_arr,
        "BER_QPSK":        ber_qpsk_arr,
        "BER_16QAM":       ber_qam16_arr,
        "Best_Modulation": best_mod_arr
    })
    return df


if __name__ == "__main__":
    snr_range      = np.arange(-4, 21, step=0.5)
    trials_per_snr = 10
    N_per_trial    = 100_000

    df = generate_dataset(snr_range, trials_per_snr, N_per_trial)
    df.to_csv("modulation_dataset.csv", index=False)

    print(f"Dataset saved: {len(df)} rows")
    print("\nFirst few rows:")
    print(df.head(10).to_string(index=False))
    print("\nLabel distribution:")
    print(df["Best_Modulation"].value_counts())
    print("\nSNR vs Best Modulation:")
    print(df.groupby("SNR_dB")["Best_Modulation"].first().to_string())