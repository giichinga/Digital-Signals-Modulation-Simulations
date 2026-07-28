import numpy as np # type: ignore[import-not-found]
import matplotlib.pyplot as plt # type: ignore[import-not-found]
from scipy.special import erfc # type: ignore[import-not-found]

# ── Import everything──────────────────────────
from BPSK  import bpsk_modulate, bpsk_demodulate, compute_ber, theoretical_ber_bpsk, add_awgn_noise as awgn_bpsk
from QPSK  import qpsk_modulate, qpsk_demodulate, theoretical_ber_qpsk, add_awgn_noise as awgn_qpsk
from QAM16 import qam16_modulate, qam16_demodulate, theoretical_ber_qam16, add_awgn_noise as awgn_qam16


# ── Simulation parameters ──────────────────────────────────────────────────
EbN0_dB = np.arange(-20, 21, step=1)   # -20 dB to 20 dB covers all three schemes

# Adaptive simulation controls: instead of a single fixed N for every SNR
# point (wasteful at low SNR, unreliable at high SNR where errors are
# rare), keep simulating in batches per SNR point until enough bit errors
# have been observed to trust the BER estimate, up to a hard cap.
BATCH_BITS = 500_000
MIN_ERRORS = 100
MAX_BITS = 50_000_000

# Storage for simulated BER values
BER_bpsk  = np.zeros(len(EbN0_dB))
BER_qpsk  = np.zeros(len(EbN0_dB))
BER_qam16 = np.zeros(len(EbN0_dB))


def simulate_ber_adaptive(modulate, demodulate, awgn, snr_db, bits_multiple=1,
                           batch_bits=BATCH_BITS, min_errors=MIN_ERRORS,
                           max_bits=MAX_BITS):
    """
    Run bit batches through modulate -> awgn -> demodulate, accumulating
    bits/errors until either min_errors errors have been observed or
    max_bits total bits have been simulated (whichever comes first).
    """
    n = batch_bits - (batch_bits % bits_multiple)

    total_bits = 0
    total_errors = 0

    while total_errors < min_errors and total_bits < max_bits:
        bits = np.random.randint(0, 2, n)
        symbols = modulate(bits)
        received = awgn(symbols, snr_db)
        bits_hat = demodulate(received)

        total_errors += int(np.sum(bits != bits_hat))
        total_bits += n

    return total_errors / total_bits, total_bits


# ── Run all three simulations ──────────────────────────────────────────────
print("=" * 60)
print("  BPSK vs QPSK vs 16-QAM — BER Comparison")
print("=" * 60)
print(f"  Adaptive simulation: batches of {BATCH_BITS:,} bits,")
print(f"  stopping at {MIN_ERRORS} errors (cap {MAX_BITS:,} bits/point)")
print(f"  SNR range          : {EbN0_dB[0]} dB to {EbN0_dB[-1]} dB")
print("-" * 60)
print(f"  {'Eb/N0':>6}  {'BPSK BER':>12}  {'QPSK BER':>12}  {'16-QAM BER':>12}")
print("-" * 60)

for i, snr in enumerate(EbN0_dB):

    # ── BPSK (1 bit/symbol, real-valued) ────────────────────────────────
    BER_bpsk[i], _ = simulate_ber_adaptive(
        bpsk_modulate, bpsk_demodulate, awgn_bpsk, snr, bits_multiple=1
    )

    # ── QPSK (2 bits/symbol) ─────────────────────────────────────────────
    BER_qpsk[i], _ = simulate_ber_adaptive(
        qpsk_modulate, qpsk_demodulate, awgn_qpsk, snr, bits_multiple=2
    )

    # ── 16-QAM (4 bits/symbol) ───────────────────────────────────────────
    BER_qam16[i], _ = simulate_ber_adaptive(
        qam16_modulate, qam16_demodulate, awgn_qam16, snr, bits_multiple=4
    )

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
ax.set_xlim([-20, 20])

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