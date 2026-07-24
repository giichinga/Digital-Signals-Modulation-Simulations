import numpy as np # type: ignore[import-not-found]
import matplotlib.pyplot as plt # type: ignore[import-not-found]
from scipy.special import erfc # type: ignore[import-not-found]

# ── Import everything──────────────────────────
from BPSK  import bpsk_modulate, bpsk_demodulate, compute_ber, theoretical_ber_bpsk
from QPSK  import qpsk_modulate, qpsk_demodulate, theoretical_ber_qpsk, add_awgn_noise as awgn_qpsk
from QAM16 import qam16_modulate, qam16_demodulate, theoretical_ber_qam16, add_awgn_noise as awgn_qam16


EbN0_dB = 5
EbN0 = 10**(EbN0_dB/10)

# BPSK check
bits = np.random.randint(0, 2, 1000000)
symbols_bpsk = 2*bits - 1
Es_bpsk = np.mean(np.abs(symbols_bpsk)**2)
print(f"BPSK  Es={Es_bpsk}, k=1, Eb={Es_bpsk/1}, sigma={np.sqrt((Es_bpsk/1/EbN0)/2):.6f}")

# QPSK check
symbols_qpsk = (2*bits[0::2]-1) + 1j*(2*bits[1::2]-1)
Es_qpsk = np.mean(np.abs(symbols_qpsk)**2)
print(f"QPSK  Es={Es_qpsk}, k=2, Eb={Es_qpsk/2}, sigma={np.sqrt((Es_qpsk/2/EbN0)/2):.6f}")