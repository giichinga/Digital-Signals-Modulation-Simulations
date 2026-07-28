import numpy as np # type: ignore[import-not-found]
import matplotlib.pyplot as plt # type: ignore[import-not-found]
from scipy.special import erfc # type: ignore[import-not-found]

from BPSK import compute_ber
from sim_utils import simulate_ber_adaptive


def qpsk_modulate(bits):
    # Step 1: split bits into I and Q streams
    I_bits = bits[0::2]
    Q_bits = bits[1::2]
    
    # Step 2: BPSK-modulate each stream independently
    I_symbols = 2 * I_bits - 1
    Q_symbols = 2 * Q_bits - 1
    
    # Step 3: combine into complex symbols
    symbols = I_symbols + 1j * Q_symbols
    return symbols
def add_awgn_noise(symbols, EbN0_dB):
    EbN0 = 10**(EbN0_dB/10)

    Es = np.mean(np.abs(symbols)**2)   # = 2
    k = 2                              # bits/symbol
    Eb = Es / k

    N0 = Eb / EbN0
    sigma = np.sqrt(N0/2)

    noise = sigma * (
        np.random.randn(*symbols.shape)
        + 1j*np.random.randn(*symbols.shape)
    )

    return symbols + noise

def qpsk_transmit(bits, EbN0_dB):
    symbols = qpsk_modulate(bits)
    received = add_awgn_noise(symbols, EbN0_dB)
    return symbols, received


def qpsk_demodulate(received):

    I_received = np.real(received)
    Q_received = np.imag(received)
    I_bits = (I_received > 0).astype(int)
    Q_bits = (Q_received > 0).astype(int)
    return np.column_stack([I_bits, Q_bits]).flatten()

def theoretical_ber_qpsk(EbN0_dB_range):
    EbN0_linear = 10 ** (EbN0_dB_range / 10)
    return 0.5 * erfc(np.sqrt(EbN0_linear))
    

def run_qpsk_simulation():

    EbN0_dB   = np.arange(-20, 21, step=1)
    BER_sim   = np.zeros(len(EbN0_dB))

    print("=" * 60)
    print("  QPSK Simulation over AWGN Channel  (adaptive bit count)")
    print("=" * 60)
    print(f"  SNR range : {EbN0_dB[0]} dB to {EbN0_dB[-1]} dB")
    print("-" * 60)
    print(f"  {'Eb/N0':>7}  {'Sim BER':>12}  {'Theory BER':>12}  {'errors':>7}  {'bits':>12}")
    print("-" * 60)

    for i, snr in enumerate(EbN0_dB):
        ber, total_bits, total_errors = simulate_ber_adaptive(
            qpsk_modulate, qpsk_demodulate, add_awgn_noise, snr,
            bits_multiple=2,   # QPSK needs an even number of bits per symbol
        )
        BER_sim[i] = ber
        theory_val = 0.5 * erfc(np.sqrt(10 ** (snr / 10)))
        print(f"  {snr:>7.1f}  {ber:>12.3e}  {theory_val:>12.3e}  {total_errors:>7d}  {total_bits:>12,}")

    # Clip theory curve so float underflow at high SNR doesn't distort
    # the auto-scaled axis, and set explicit limits matching what the
    # adaptive bit budget can actually resolve.
    BER_theory = np.clip(theoretical_ber_qpsk(EbN0_dB), 1e-300, None)

    print("-" * 60)
    print("  Simulation complete. Plotting...")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(EbN0_dB, BER_sim,    'bo-', label='Simulated BER',   markersize=6)
    ax.semilogy(EbN0_dB, BER_theory, 'r--', label='Theoretical BER', linewidth=2)
    ax.set_xlabel('Eb/N0 (dB)',           fontsize=12)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax.set_title('QPSK Performance over AWGN Channel', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.set_ylim([1e-7, 1])
    ax.set_xlim([EbN0_dB[0], EbN0_dB[-1]])
    plt.tight_layout()
    plt.savefig('qpsk_ber_curve.png', dpi=150)
    print("  Plot saved as: qpsk_ber_curve.png")
    plt.show()
    

if __name__ == "__main__":
    run_qpsk_simulation()