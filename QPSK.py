import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

def qpsk_modulate(bits):
    # Step 1: split bits into I and Q streams
    I_bits = bits[0::2]
    Q_bits = bits[1::2]
    
    # Step 2: BPSK-modulate each stream independently
    # (map 0 → -1, 1 → +1)
    I_symbols = 2 * I_bits - 1
    Q_symbols = 2 * Q_bits - 1
    
    # Step 3: combine into complex symbols
    # I is the real part, Q is the imaginary part
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

def qpsk_demodulate(received):

    I_received = np.real(received)
    Q_received = np.imag(received)
    I_bits = (I_received > 0).astype(int)
    Q_bits = (Q_received > 0).astype(int)
    return np.column_stack([I_bits, Q_bits]).flatten()

def compute_ber(bits_sent, bits_received):
    errors = np.sum(bits_sent != bits_received)
    return errors / len(bits_sent)

def theoretical_ber_qpsk(EbN0_dB_range):
    EbN0_linear = 10 ** (EbN0_dB_range / 10)
    return 0.5 * erfc(np.sqrt(EbN0_linear))
    

def run_qpsk_simulation():
    
    N         = 5_000_000
    EbN0_dB   = np.arange(-4, 11, step=1)        # arange not arrange
    BER_sim   = np.zeros(len(EbN0_dB))
    
    
    print("=" * 50)
    print("  QPSK Simulation over AWGN Channel")
    print("=" * 50)
    print(f"  Bits per SNR point : {N:,}")
    print(f"  SNR range          : {EbN0_dB[0]} dB to {EbN0_dB[-1]} dB")
    print("-" * 50)
    print(f"  {'Eb/N0 (dB)':>12}  {'Simulated BER':>15}  {'Theoretical BER':>16}")
    print("-" * 50)
    
    for i, snr in enumerate(EbN0_dB):
        bits        = np.random.randint(0, 2, N)  # N bits, but we need pairs for QPSK
        symbols     = qpsk_modulate(bits)
        received    = add_awgn_noise(symbols, snr)
        bits_hat    = qpsk_demodulate(received)
        BER_sim[i]  = compute_ber(bits, bits_hat)
        theory_val  = 0.5 * erfc(np.sqrt(10 ** (snr / 10)))
        print(f"  {snr:>12.1f}  {BER_sim[i]:>15.6f}  {theory_val:>16.6f}")

        # computed AFTER the loop, not inside it
    
    BER_theory = theoretical_ber_qpsk(EbN0_dB)
    
    print("-" * 50)
    print("  Simulation complete. Plotting...")
    
        # --- Plot --- all inside the function, so variables are in scope 
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(EbN0_dB, BER_sim,    'bo-', label='Simulated BER',   markersize=6)
    ax.semilogy(EbN0_dB, BER_theory, 'r--', label='Theoretical BER', linewidth=2)
    ax.set_xlabel('Eb/N0 (dB)',           fontsize=12)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax.set_title('QPSK Performance over AWGN Channel', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.set_ylim([1e-5, 1])
    ax.set_xlim([-4, 10])
    plt.tight_layout()
    plt.savefig('qpsk_ber_curve.png', dpi=150)
    print("  Plot saved as: qpsk_ber_curve.png")
    plt.show()
    

if __name__ == "__main__":
    run_qpsk_simulation()