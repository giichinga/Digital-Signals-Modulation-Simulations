import numpy as np  # type: ignore[import-not-found]
import matplotlib.pyplot as plt  # type: ignore[import-not-found]
from scipy.special import erfc  # type: ignore[import-not-found]

from sim_utils import simulate_ber_adaptive


def bpsk_modulate(bits):
    symbols = 2 * bits - 1
    return symbols


def add_awgn_noise(symbols, EbN0_dB):
    EbN0 = 10 ** (EbN0_dB / 10)

    Es = np.mean(np.abs(symbols) ** 2)
    k = 1
    Eb = Es / k
    N0 = Eb / EbN0

    sigma = np.sqrt(N0 / 2)

    noise = sigma * np.random.randn(len(symbols))

    return symbols + noise


# -------------------------------------------------
# NEW FUNCTION FOR MACHINE LEARNING DATASET
# -------------------------------------------------
def bpsk_transmit(bits, EbN0_dB):
    """
    Modulate bits and pass through AWGN channel.

    Returns
    -------
    symbols : transmitted symbols
    received : noisy received symbols
    """
    symbols = bpsk_modulate(bits)
    received = add_awgn_noise(symbols, EbN0_dB)
    return symbols, received


def bpsk_demodulate(received):
    return (received > 0).astype(int)


def compute_ber(bits_sent, bits_received):
    errors = np.sum(bits_sent != bits_received)
    return errors / len(bits_sent)


def theoretical_ber_bpsk(EbN0_dB_range):
    EbN0_linear = 10 ** (EbN0_dB_range / 10)
    return 0.5 * erfc(np.sqrt(EbN0_linear))


def run_bpsk_simulation():

    EbN0_dB = np.arange(-20, 21, 1)

    BER_sim   = np.zeros(len(EbN0_dB))
    bits_used = np.zeros(len(EbN0_dB), dtype=np.int64)

    print("=" * 60)
    print("BPSK Simulation  (adaptive bit count per SNR point)")
    print("=" * 60)

    for i, snr in enumerate(EbN0_dB):

        ber, total_bits, total_errors = simulate_ber_adaptive(
            bpsk_modulate, bpsk_demodulate, add_awgn_noise, snr,
            bits_multiple=1,
        )
        BER_sim[i]   = ber
        bits_used[i] = total_bits

        theory = 0.5 * erfc(np.sqrt(10 ** (snr / 10)))

        print(
            f"SNR={snr:5.1f} dB   "
            f"BER={ber:.3e}   "
            f"Theory={theory:.3e}   "
            f"errors={total_errors:4d}   "
            f"bits={total_bits:,}"
        )

    # Theory curve underflows to exactly 0 for high Eb/N0 (float64 can't
    # represent values below ~1e-308) - clip so log-scale plotting and
    # any downstream log() calls don't choke on it.
    BER_theory = np.clip(theoretical_ber_bpsk(EbN0_dB), 1e-300, None)

    # BER_sim can be exactly 0 where no errors were observed even after
    # hitting the max_bits cap - matplotlib's semilogy just breaks the
    # line at those points (log(0) undefined), which is the CORRECT
    # behaviour: it honestly shows "no error observed", not a fabricated
    # near-zero value.
    plt.figure(figsize=(8, 5))

    plt.semilogy(EbN0_dB, BER_sim, "bo-", label="Simulation")
    plt.semilogy(EbN0_dB, BER_theory, "r--", label="Theory")

    plt.grid(True, which="both")
    plt.legend()
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("BER")
    plt.title("BPSK over AWGN")

    # Explicit, sane y-limits instead of letting matplotlib auto-scale
    # to the theory curve's underflow tail.
    plt.ylim(1e-7, 1)
    plt.xlim(EbN0_dB[0], EbN0_dB[-1])

    plt.tight_layout()
    plt.savefig("bpsk_ber_curve.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    run_bpsk_simulation()