"""
comparison.py
=============
Combines BPSK, QPSK and 16-QAM simulations on one plot.
This is your first major project deliverable.

Run it:
    python comparison.py

Make sure BPSK.py, QPSK.py and QAM16.py are in the same folder.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# ── Import everything we already wrote and tested ──────────────────────────
from BPSK  import bpsk_modulate, bpsk_demodulate, compute_ber, theoretical_ber_bpsk
from QPSK  import qpsk_modulate, qpsk_demodulate, theoretical_ber_qpsk, add_awgn_noise as awgn_qpsk
from QAM16 import qam16_modulate, qam16_demodulate, theoretical_ber_qam16, add_awgn_noise as awgn_qam16


# ── Simulation parameters ──────────────────────────────────────────────────
N       = 5_000_000                       # bits per SNR point
N       = N - (N % 4)                  # ensure divisible by 4 for 16-QAM
EbN0_dB = np.arange(-4, 21, step=1)   # -4 dB to 20 dB covers all three schemes

# Storage for simulated BER values
BER_bpsk  = np.zeros(len(EbN0_dB))
BER_qpsk  = np.zeros(len(EbN0_dB))
BER_qam16 = np.zeros(len(EbN0_dB))


# ── BPSK noise function (real-valued, not complex) ─────────────────────────
def awgn_bpsk(symbols, EbN0_dB):
    EbN0_linear = 10 ** (EbN0_dB / 10)
    noise_std   = np.sqrt(1 / (2 * EbN0_linear))
    return symbols + noise_std * np.random.randn(len(symbols))


# ── Run all three simulations ──────────────────────────────────────────────
print("=" * 60)
print("  BPSK vs QPSK vs 16-QAM — BER Comparison")
print("=" * 60)
print(f"  Bits per SNR point : {N:,}")
print(f"  SNR range          : {EbN0_dB[0]} dB to {EbN0_dB[-1]} dB")
print("-" * 60)
print(f"  {'Eb/N0':>6}  {'BPSK BER':>12}  {'QPSK BER':>12}  {'16-QAM BER':>12}")
print("-" * 60)

for i, snr in enumerate(EbN0_dB):

    # ── BPSK ──────────────────────────────────────────────────────────────
    bits_bpsk        = np.random.randint(0, 2, N)
    symbols_bpsk     = bpsk_modulate(bits_bpsk)
    received_bpsk    = awgn_bpsk(symbols_bpsk, snr)
    bits_hat_bpsk    = bpsk_demodulate(received_bpsk)
    BER_bpsk[i]      = compute_ber(bits_bpsk, bits_hat_bpsk)

    # ── QPSK ──────────────────────────────────────────────────────────────
    bits_qpsk        = np.random.randint(0, 2, N)
    symbols_qpsk     = qpsk_modulate(bits_qpsk)
    received_qpsk    = awgn_qpsk(symbols_qpsk, snr)
    bits_hat_qpsk    = qpsk_demodulate(received_qpsk)
    BER_qpsk[i]      = compute_ber(bits_qpsk, bits_hat_qpsk)

    # ── 16-QAM ────────────────────────────────────────────────────────────
    bits_qam         = np.random.randint(0, 2, N)
    symbols_qam      = qam16_modulate(bits_qam)
    received_qam     = awgn_qam16(symbols_qam, snr)
    bits_hat_qam     = qam16_demodulate(received_qam)
    BER_qam16[i]     = compute_ber(bits_qam, bits_hat_qam)

    print(f"  {snr:>6.1f}  {BER_bpsk[i]:>12.6f}  {BER_qpsk[i]:>12.6f}  {BER_qam16[i]:>12.6f}")

print("-" * 60)
print("  All simulations complete. Plotting...")


# ── Theoretical curves ─────────────────────────────────────────────────────
BER_bpsk_theory  = theoretical_ber_bpsk(EbN0_dB)
BER_qpsk_theory  = theoretical_ber_qpsk(EbN0_dB)
BER_qam16_theory = theoretical_ber_qam16(EbN0_dB)


# ── Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

# BPSK — blue
ax.semilogy(EbN0_dB, BER_bpsk,        'b^-',  label='BPSK (simulated)',      markersize=5)
ax.semilogy(EbN0_dB, BER_bpsk_theory, 'b--',  label='BPSK (theoretical)',    linewidth=1.5)

# QPSK — green
ax.semilogy(EbN0_dB, BER_qpsk,        'gs-',  label='QPSK (simulated)',      markersize=5)
ax.semilogy(EbN0_dB, BER_qpsk_theory, 'g--',  label='QPSK (theoretical)',    linewidth=1.5)

# 16-QAM — red
ax.semilogy(EbN0_dB, BER_qam16,       'ro-',  label='16-QAM (simulated)',    markersize=5)
ax.semilogy(EbN0_dB, BER_qam16_theory,'r--',  label='16-QAM (theoretical)',  linewidth=1.5)

# ── Annotation: explain the SNR gap ───────────────────────────────────────
# Draw a horizontal reference line at BER = 1e-3
ax.axhline(y=1e-3, color='gray', linestyle=':', linewidth=1, alpha=0.7)
ax.text(14, 1.5e-3, 'BER = 10⁻³ reference', fontsize=9, color='gray')

ax.set_xlabel('Eb/N0 (dB)',            fontsize=12)
ax.set_ylabel('Bit Error Rate (BER)',  fontsize=12)
ax.set_title('BPSK vs QPSK vs 16-QAM — BER Performance over AWGN Channel', fontsize=13)
ax.legend(fontsize=10, loc='lower left')
ax.grid(True, which='both', linestyle='--', alpha=0.5)
ax.set_ylim([1e-5, 1])
ax.set_xlim([-4, 20])

# ── Key insight printed to terminal ───────────────────────────────────────
print()
print("=" * 60)
print("  KEY RESULTS SUMMARY")
print("=" * 60)
print("  At BER = 1e-3 (1 error per 1000 bits):")
print()
print("  BPSK  : ~7 dB SNR needed  | 1 bit/symbol")
print("  QPSK  : ~7 dB SNR needed  | 2 bits/symbol  (2x throughput, no penalty)")
print("  16-QAM: ~13 dB SNR needed | 4 bits/symbol  (4x throughput, +6 dB cost)")
print()
print("  This SNR gap is exactly what the neural network will learn to exploit.")
print("  Low SNR  → use BPSK  (robust)")
print("  Mid SNR  → use QPSK  (efficient)")
print("  High SNR → use 16-QAM (maximum throughput)")
print("=" * 60)

plt.tight_layout()
plt.savefig('comparison_ber.png', dpi=150)
print("  Plot saved as: comparison_ber.png")
plt.show()


