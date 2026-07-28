from BPSK import compute_ber
import numpy as np # type: ignore[import-not-found]
import matplotlib.pyplot as plt # type: ignore[import-not-found]
from scipy.special import erfc # type: ignore[import-not-found]

from sim_utils import simulate_ber_adaptive

def qam16_modulate(bits):
    symbols = []
    for i in range(0, len(bits), 4):
        b0, b1, b2, b3 = bits[i:i+4]
        I = gray_to_amplitude(b0, b1)
        Q = gray_to_amplitude(b2, b3)
        symbols.append(I + 1j * Q)
    return np.array(symbols)
    

def gray_to_amplitude(b0, b1):
    # Map 2 bits to one of {-3, -1, +1, +3} using true Gray order
    # (00, 01, 11, 10) so a one-level amplitude slip flips only 1 bit.
    if b0 == 0 and b1 == 0:
        return -3
    elif b0 == 0 and b1 == 1:
        return -1
    elif b0 == 1 and b1 == 1:
        return +1
    elif b0 == 1 and b1 == 0:
        return +3
    

def add_awgn_noise(symbols, EbN0_dB):
    EbN0 = 10**(EbN0_dB/10)

    Es = np.mean(np.abs(symbols)**2)   # = 10 for 16-QAM
    k = 4                              # bits/symbol
    Eb = Es / k

    N0 = Eb / EbN0
    sigma = np.sqrt(N0/2)

    noise = sigma * (
        np.random.randn(*symbols.shape)
        + 1j*np.random.randn(*symbols.shape)
    )
    
    return symbols + noise
    
def qam16_transmit(bits, EbN0_dB):
    symbols = qam16_modulate(bits)
    received = add_awgn_noise(symbols, EbN0_dB)
    return symbols, received  

def qam16_demodulate(received):
    # Decide each axis independently using 3 thresholds: -2, 0, +2

    I_received = np.real(received)
    Q_received = np.imag(received)
    I_bits = np.zeros((len(I_received), 2), dtype=int)
    Q_bits = np.zeros((len(Q_received), 2), dtype=int)
    
    I_bits[I_received < -2] = [0, 0]
    I_bits[(I_received >= -2) & (I_received < 0)] = [0, 1]
    I_bits[(I_received >= 0) & (I_received < 2)] = [1, 1]
    I_bits[I_received >= 2] = [1, 0]
    Q_bits[Q_received < -2] = [0, 0]
    Q_bits[(Q_received >= -2) & (Q_received < 0)] = [0, 1]
    Q_bits[(Q_received >= 0) & (Q_received < 2)] = [1, 1]
    Q_bits[Q_received >= 2] = [1, 0]
    
    return np.column_stack([I_bits, Q_bits]).flatten()

def amplitude_to_bits(amplitudes):
    # Reverse of gray_to_amplitude
    # Map {-3,-1,+1,+3} back to 2-bit pairs (matches Gray order above)

    bits = []
    for amp in amplitudes:
        if amp == -3:
            bits.extend([0, 0])
        elif amp == -1:
            bits.extend([0, 1])
        elif amp == +1:
            bits.extend([1, 1])
        elif amp == +3:
            bits.extend([1, 0])
    return np.array(bits)


def theoretical_ber_qam16(EbN0_dB_range):
    EbN0_linear = 10 ** (EbN0_dB_range / 10)
    # Gray-coded 16-QAM: BER = (3/8) * erfc(sqrt(0.4 * Eb/N0))
    return 0.375 * erfc(np.sqrt(0.4 * EbN0_linear))

def run_qam16_simulation():

    EbN0_dB   = np.arange(-20, 21, step=1)
    BER_sim   = np.zeros(len(EbN0_dB))

    print("=" * 60)
    print("  16-QAM Simulation over AWGN Channel  (adaptive bit count)")
    print("=" * 60)
    print(f"  SNR range : {EbN0_dB[0]} dB to {EbN0_dB[-1]} dB")
    print("-" * 60)
    print(f"  {'Eb/N0':>7}  {'Sim BER':>12}  {'Theory BER':>12}  {'errors':>7}  {'bits':>12}")
    print("-" * 60)

    for i, snr in enumerate(EbN0_dB):
        ber, total_bits, total_errors = simulate_ber_adaptive(
            qam16_modulate, qam16_demodulate, add_awgn_noise, snr,
            bits_multiple=4,   # 16-QAM needs bits in groups of 4
        )
        BER_sim[i] = ber
        theory_val = theoretical_ber_qam16(snr)
        print(f"  {snr:>7.1f}  {ber:>12.3e}  {theory_val:>12.3e}  {total_errors:>7d}  {total_bits:>12,}")

    BER_theory = np.clip(theoretical_ber_qam16(EbN0_dB), 1e-300, None)

    print("-" * 60)
    print("  Simulation complete. Plotting...")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(EbN0_dB, BER_sim,    'bo-', label='Simulated BER',   markersize=6)
    ax.semilogy(EbN0_dB, BER_theory, 'r--', label='Theoretical BER', linewidth=2)
    ax.set_xlabel('Eb/N0 (dB)',           fontsize=12)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax.set_title('16-QAM Performance over AWGN Channel', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.set_ylim([1e-7, 1])
    ax.set_xlim([EbN0_dB[0], EbN0_dB[-1]])   # was mismatched [-4,20] before - fixed to match the actual sweep
    plt.tight_layout()
    plt.savefig('qam16_ber_curve.png', dpi=150)
    print("  Plot saved as: qam16_ber_curve.png")
    plt.show()


if __name__ == "__main__":
    run_qam16_simulation()