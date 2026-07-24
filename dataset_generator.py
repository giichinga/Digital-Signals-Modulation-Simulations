import numpy as np # type: ignore[import-not-found]
import pandas as pd # type: ignore[import-not-found]
from BPSK  import bpsk_modulate, bpsk_demodulate, compute_ber
from QPSK  import qpsk_modulate, qpsk_demodulate
from QAM16 import qam16_modulate, qam16_demodulate

# Import noise functions from each file
from BPSK  import add_awgn_noise as awgn_bpsk
from QPSK  import add_awgn_noise as awgn_qpsk
from QAM16 import add_awgn_noise as awgn_qam16

BER_THRESHOLD = 1e-2
TRIALS = 10
BITS_PER_TRIAL = 100_000 

rng = np.random.default_rng(42)

# Mapping for readability :
# 0 = BPSK
# 1 = QPSK
# 2 = 16-QAM

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
    dataset_rows = []
    for snr_db in snr_range:
        for trial in range(trials_per_snr):
            # Simulate BPSK
            bits_bpsk = rng.integers(0, 2, N_per_trial)
            ber_bpsk = simulate(bpsk_modulate, bpsk_demodulate, awgn_bpsk, bits_bpsk, snr_db)

            # Simulate QPSK
            bits_qpsk = rng.integers(0, 2, N_per_trial)
            ber_qpsk = simulate(qpsk_modulate, qpsk_demodulate, awgn_qpsk, bits_qpsk, snr_db)

            # Simulate 16-QAM
            bits_qam16 = rng.integers(0, 2, N_per_trial)
            ber_qam16 = simulate(qam16_modulate, qam16_demodulate, awgn_qam16, bits_qam16, snr_db)

            # Determine best modulation scheme
            best_modulation = get_best_modulation(ber_bpsk, ber_qpsk, ber_qam16)

            # Append row to dataset
            dataset_rows.append({
                "SNR_dB": snr_db,
                "Trial": trial + 1,
                "BER_BPSK": ber_bpsk,
                "BER_QPSK": ber_qpsk,
                "BER_16QAM": ber_qam16,
                "Best_Modulation": best_modulation
            })
    return dataset_rows

if __name__ == "__main__":
    # SNR range covering all three modulation regimes
    snr_range = np.arange(-4, 21, step=0.5)  # finer steps than before
    
    trials_per_snr = 10   # 10 rows per SNR value
    N_per_trial    = 100_000  # bits per trial
    
    rows = generate_dataset(snr_range, trials_per_snr, N_per_trial)
    
    df = pd.DataFrame(rows)
    df.to_csv("modulation_dataset.csv", index=False)
    
    print(f"Dataset saved: {len(rows)} rows")
    print("\nFirst few rows:")
    print(df.head(10).to_string(index=False))   
    
