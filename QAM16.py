
from BPSK import compute_ber
import numpy as np # type: ignore[import-not-found]
import matplotlib.pyplot as plt # type: ignore[import-not-found]
from scipy.special import erfc # type: ignore[import-not-found]

def qam16_modulate(bits):
    # 16-QAM: 4 bits per symbol
    # Split into groups of 4: first 2 bits → I, last 2 bits → Q
    # Each 2-bit pair maps to one of {-3, -1, +1, +3}
    # This is the PAM-4 mapping on each axis
    #
    # Mapping: 00 → -3, 01 → -1, 11 → +1, 10 → +3  (Gray coded)
    # For simplicity, use: 00→-3, 01→-1, 10→+1, 11→+3
    #
    # Hint: reshape bits into groups of 4
    # then split each group into 2-bit I and 2-bit Q pairs
    # then map each 2-bit pair to {-3,-1,+1,+3}
    symbols = []
    for i in range(0, len(bits), 4):
        b0, b1, b2, b3 = bits[i:i+4]
        I = gray_to_amplitude(b0, b1)
        Q = gray_to_amplitude(b2, b3)
        symbols.append(I + 1j * Q)
    return np.array(symbols)
    pass

def gray_to_amplitude(b0, b1):
    # Map 2 bits to one of {-3, -1, +1, +3}
    # b0 is MSB, b1 is LSB
    # 00 → -3
    # 01 → -1
    # 10 → +1  (note: not Gray coded, just natural binary for now)
    # 11 → +3
    # YOUR CODE HERE
    if b0 == 0 and b1 == 0:
        return -3
    elif b0 == 0 and b1 == 1:
        return -1
    elif b0 == 1 and b1 == 0:
        return +1
    elif b0 == 1 and b1 == 1:
        return +3
    pass

def add_awgn_noise(symbols, EbN0_dB):
    # Same as QPSK — complex noise on both I and Q
    # BUT: 16-QAM average symbol energy is not 1
    # Average energy of {-3,-1,+1,+3} = (9+1+1+9)/4 = 5 per axis
    # So average symbol energy Es = 10
    # bits per symbol k = 4
    # Therefore: noise_std = sqrt(Es / (2 * k * EbN0_linear))
    #                      = sqrt(10 / (8 * EbN0_linear))
    # YOUR CODE HERE
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
    
    

def qam16_demodulate(received):
    # Decide each axis independently using 3 thresholds: -2, 0, +2
    # If real(r) > 2  → +3 → bits 11
    # If 0 < real(r) < 2 → +1 → bits 10
    # If -2 < real(r) < 0 → -1 → bits 01
    # If real(r) < -2 → -3 → bits 00
    # Same logic for imaginary axis
    # YOUR CODE HERE
    I_received = np.real(received)
    Q_received = np.imag(received)
    I_bits = np.zeros((len(I_received), 2), dtype=int)
    Q_bits = np.zeros((len(Q_received), 2), dtype=int)
    
    I_bits[I_received < -2] = [0, 0]
    I_bits[(I_received >= -2) & (I_received < 0)] = [0, 1]
    I_bits[(I_received >= 0) & (I_received < 2)] = [1, 0]
    I_bits[I_received >= 2] = [1, 1]
    Q_bits[Q_received < -2] = [0, 0]
    Q_bits[(Q_received >= -2) & (Q_received < 0)] = [0, 1]
    Q_bits[(Q_received >= 0) & (Q_received < 2)] = [1, 0]
    Q_bits[Q_received >= 2] = [1, 1]
    
    return np.column_stack([I_bits, Q_bits]).flatten()

def amplitude_to_bits(amplitudes):
    # Reverse of gray_to_amplitude
    # Map {-3,-1,+1,+3} back to 2-bit pairs
    # YOUR CODE HERE
    bits = []
    for amp in amplitudes:
        if amp == -3:
            bits.extend([0, 0])
        elif amp == -1:
            bits.extend([0, 1])
        elif amp == +1:
            bits.extend([1, 0])
        elif amp == +3:
            bits.extend([1, 1])
    return np.array(bits)
    pass

def theoretical_ber_qam16(EbN0_dB_range):
    # BER = 0.75 * erfc(sqrt(0.4 * Eb/N0))
    # YOUR CODE HERE
    EbN0_linear = 10 ** (EbN0_dB_range / 10)
    return 0.75 * erfc(np.sqrt(0.4 * EbN0_linear))
    pass

def run_qam16_simulation():
    # Same loop structure as BPSK and QPSK
    # Remember: N bits → N/4 symbols (4 bits per symbol)
    # YOUR CODE HERE
    N         = 5_000_000
    N = N - (N % 4)   # ensure N is always divisible by 4
    EbN0_dB   = np.arange(-4, 21, step=1)        # arange not arrange
    BER_sim   = np.zeros(len(EbN0_dB))
    
    
    print("=" * 50)
    print("  QAM16 Simulation over AWGN Channel")
    print("=" * 50)
    print(f"  Bits per SNR point : {N:,}")
    print(f"  SNR range          : {EbN0_dB[0]} dB to {EbN0_dB[-1]} dB")
    print("-" * 50)
    print(f"  {'Eb/N0 (dB)':>12}  {'Simulated BER':>15}  {'Theoretical BER':>16}")
    print("-" * 50)
    
    for i, snr in enumerate(EbN0_dB):
        bits        = np.random.randint(0, 2, N)  
        symbols     = qam16_modulate(bits)
        received    = add_awgn_noise(symbols, snr)
        bits_hat    = qam16_demodulate(received)
        BER_sim[i]  = compute_ber(bits, bits_hat)
        theory_val  = theoretical_ber_qam16(snr)
        print(f"  {snr:>12.1f}  {BER_sim[i]:>15.6f}  {theory_val:>16.6f}")

        # computed AFTER the loop, not inside it
    
    BER_theory = theoretical_ber_qam16(EbN0_dB)
    
    print("-" * 50)
    print("  Simulation complete. Plotting...")
    
        # --- Plot --- all inside the function, so variables are in scope 
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(EbN0_dB, BER_sim,    'bo-', label='Simulated BER',   markersize=6)
    ax.semilogy(EbN0_dB, BER_theory, 'r--', label='Theoretical BER', linewidth=2)
    ax.set_xlabel('Eb/N0 (dB)',           fontsize=12)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax.set_title('QAM16 Performance over AWGN Channel', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.set_ylim([1e-5, 1])
    ax.set_xlim([-4, 20])
    plt.tight_layout()
    plt.savefig('qam16_ber_curve.png', dpi=150)
    print("  Plot saved as: qam16_ber_curve.png")
    plt.show()
    pass

if __name__ == "__main__":
    run_qam16_simulation()