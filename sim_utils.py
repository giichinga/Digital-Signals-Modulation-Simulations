"""
sim_utils.py
============
Shared helper for running BER simulations with an ADAPTIVE bit count.

Why this exists:
-----------------
A single fixed N (e.g. 5,000,000 bits) cannot correctly cover a wide SNR
sweep. At low SNR you're wasting time simulating far more bits than you
need. At high SNR you eventually can't observe ANY errors at all, because
the smallest measurable BER with N bits is 1/N - once the true BER drops
below that floor, BER_sim becomes exactly 0, which breaks on a log-scale
plot (log(0) is undefined) and also can't be trusted as "the true BER is
zero" - it just means you didn't simulate enough bits to catch a rare
event.

The fix: keep simulating in batches for a given SNR point until either
enough bit errors have been observed to trust the estimate, or a hard
cap on total bits is reached (so it doesn't run forever at very high SNR
where errors become vanishingly rare).
"""

import numpy as np


def simulate_ber_adaptive(modulate, demodulate, awgn, snr_db,
                           batch_bits=500_000, min_errors=100,
                           max_bits=50_000_000, bits_multiple=1):
    """
    Run bit batches through modulate -> awgn -> demodulate, accumulating
    bits/errors until either min_errors errors have been observed or
    max_bits total bits have been simulated (whichever comes first).

    bits_multiple: batch size is rounded down to a multiple of this so
    modulators that group bits (QPSK=2, 16-QAM=4) never receive a
    leftover partial group.
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

    ber = total_errors / total_bits
    return ber, total_bits, total_errors