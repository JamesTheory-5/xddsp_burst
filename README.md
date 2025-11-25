# xddsp_burst

```text
MODULE NAME:
xddsp_burst

DESCRIPTION:
Decaying, bandpass-filtered noise burst with an exponentially decaying amplitude envelope
and a time-varying center frequency that sweeps between two frequencies as the envelope decays.
Designed in a pure functional NumPy + Numba style with tuple-based state and block processing.

INPUTS:
- x[n]             : white noise input signal (1-D NumPy array, shape (N,))
- sr_hz            : sample rate in Hz (scalar float, fixed per module instance)
- freq_start_hz    : starting center frequency in Hz (scalar float)
- freq_end_hz      : ending center frequency in Hz (scalar float)
- decay_time_sec   : time (seconds) for the envelope to decay by 60 dB (scalar float)
- q                : bandpass filter quality factor (scalar float)
- amp              : initial envelope amplitude at start of burst (scalar float)
- smoothing_alpha  : parameter smoothing coefficient for center frequency (0..1, scalar float)
- min_env          : minimum envelope level, below which the output is effectively silent (scalar float)

OUTPUTS:
- y[n]             : decaying filtered noise burst signal (1-D NumPy array, shape (N,))
- state            : updated state tuple for continued processing

STATE VARIABLES:
(state_env,
 state_z1,
 state_z2,
 state_b0,
 state_b1,
 state_b2,
 state_a1,
 state_a2,
 state_fc_prev,
 state_decay_coeff)

EQUATIONS / MATH:

Envelope:
Let env[n] be the envelope at sample n. We define the exponential decay such that
the envelope falls by 60 dB after T = decay_time_sec seconds:

    decay_coeff = 10^(-60 / (20 * T * sr_hz))

Per-sample update:

    env[n+1] = env[n] * decay_coeff

We clamp the envelope to be non-negative:

    env[n] >= 0

Frequency sweep:
We define an "envelope-normalized" value:

    env_norm[n] = env[n] / max(env_init, eps)

where env_init = amp (initial envelope) and eps is a small positive constant to avoid division by zero.

The target center frequency is then an envelope-weighted interpolation:

    fc_target[n] = freq_end_hz + (freq_start_hz - freq_end_hz) * env_norm[n]

Parameter smoothing (first-order low-pass on frequency):

    fc_smooth[n]   = smoothing_alpha * fc_target[n] +
                     (1 - smoothing_alpha) * fc_smooth[n-1]

We store fc_smooth[n] as state_fc_prev.

Bandpass biquad (constant skirt gain, peak gain = Q form):
Given normalized angular frequency:

    w0    = 2π * fc_smooth[n] / sr_hz
    sinw0 = sin(w0)
    cosw0 = cos(w0)
    alpha = sinw0 / (2 * q)

Raw coefficients:

    b0_ =   q * alpha
    b1_ = 0.0
    b2_ = -q * alpha
    a0_ = 1.0 + alpha
    a1_ = -2.0 * cosw0
    a2_ =  1.0 - alpha

Normalize by a0_:

    b0 = b0_ / a0_
    b1 = b1_ / a0_
    b2 = b2_ / a0_
    a1 = a1_ / a0_
    a2 = a2_ / a0_

Direct Form II Transposed structure:

    # state_z1, state_z2 are the internal filter states
    y_bp[n]   = b0 * x[n] + state_z1
    state_z1' = b1 * x[n] + state_z2 - a1 * y_bp[n]
    state_z2' = b2 * x[n] - a2 * y_bp[n]

Output with envelope:

    y[n] = y_bp[n] * env[n]

state[n+1] = (env[n+1],
              state_z1',
              state_z2',
              b0, b1, b2, a1, a2,
              fc_smooth[n],
              decay_coeff)

through-zero rules:
- No through-zero modulation or phase concepts here; center frequency is clamped implicitly by math.

phase wrapping rules:
- Not applicable; biquad is defined directly in Hz, no explicit phase accumulator.

nonlinearities:
- None; system is linear in x[n] for fixed parameters. Time-varying center frequency and envelope make it LTV but still linear with respect to input.

interpolation rules:
- Frequency interpolation is done via env_norm[n], mapping the decaying envelope from 1.0 → 0.0 into freq_start_hz → freq_end_hz.
- Additional smoothing of f_c is first-order IIR with smoothing_alpha.

any time-varying coefficient rules:
- Biquad coefficients are recomputed every sample based on the smoothed center frequency fc_smooth[n].

NOTES:
- Stable operation requires:
    - sr_hz > 0
    - 0 < freq_start_hz < sr_hz / 2
    - 0 < freq_end_hz < sr_hz / 2
    - q > 0
    - 0 <= smoothing_alpha <= 1
- Very high Q and fast frequency modulation can lead to ringing; smoothing_alpha helps mitigate artifacts.
- If env decays below min_env, the output is effectively silent but the math remains stable.

```

---

## FULL PYTHON FILE: `xddsp_burst.py`

```python
"""
xddsp_burst.py

Decaying, bandpass-filtered noise burst in a pure functional NumPy + Numba style.

Behavior
--------
This module implements a burst generator that shapes an input noise signal with:

1. An exponentially decaying amplitude envelope:
   env[n+1] = env[n] * decay_coeff

   where decay_coeff is chosen such that the envelope decays by 60 dB over
   `decay_time_sec` seconds at sample rate `sr_hz`:

   decay_coeff = 10^(-60 / (20 * decay_time_sec * sr_hz))

2. A time-varying bandpass biquad whose center frequency sweeps from
   `freq_start_hz` toward `freq_end_hz` as the envelope decays:

   env_norm[n]    = env[n] / max(env_init, eps)
   fc_target[n]   = freq_end_hz + (freq_start_hz - freq_end_hz) * env_norm[n]
   fc_smooth[n]   = smoothing_alpha * fc_target[n]
                  + (1 - smoothing_alpha) * fc_smooth[n-1]

   The bandpass biquad uses the standard constant-skirt-gain, peak-gain = Q
   design based on `fc_smooth[n]` and `q`.

3. The filtered noise is then multiplied by the instantaneous envelope:

   y[n] = y_bp[n] * env[n]

Design
------
- Pure functional: no classes, no dicts.
- State is a tuple of scalars.
- All DSP functions are Numba-jitted with @njit(cache=True, fastmath=True).
- No Python objects or dynamic array allocations inside jitted code.
- Block processing is done via a jitted loop; arrays are allocated outside of jitted code.
- Parameter smoothing of the center frequency is implemented as a first-order low-pass
  on the target frequency trajectory.

Public API
----------
- xddsp_burst_init(...)
- xddsp_burst_update_state(...)
- xddsp_burst_tick(x, state, params)
- xddsp_burst_process(x, state, params)

State tuple layout
------------------
(state_env,
 state_z1,
 state_z2,
 state_b0,
 state_b1,
 state_b2,
 state_a1,
 state_a2,
 state_fc_prev,
 state_decay_coeff)

Params tuple layout
-------------------
(params_freq_start_hz,
 params_freq_end_hz,
 params_q,
 params_sr_hz,
 params_smoothing_alpha,
 params_min_env,
 params_env_init)

"""

from typing import Tuple

import numpy as np
from numba import njit


# -------------------------------------------------------------------------
# Internal helper: biquad bandpass coefficient computation
# -------------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def _xddsp_burst_compute_biquad_bandpass_coeffs(
    fc_hz: float,
    q: float,
    sr_hz: float,
) -> Tuple[float, float, float, float, float]:
    """
    Compute bandpass biquad coefficients (constant skirt gain, peak gain = Q).
    Returns normalized (b0, b1, b2, a1, a2).
    """
    # Clamp frequency to a safe range
    nyquist = 0.5 * sr_hz
    fc = fc_hz
    if fc < 1.0:
        fc = 1.0
    if fc > nyquist * 0.999:
        fc = nyquist * 0.999

    w0 = 2.0 * np.pi * fc / sr_hz
    sinw0 = np.sin(w0)
    cosw0 = np.cos(w0)

    # Avoid zero or negative Q
    q_safe = q
    if q_safe < 1e-4:
        q_safe = 1e-4

    alpha = sinw0 / (2.0 * q_safe)

    b0_ = q_safe * alpha
    b1_ = 0.0
    b2_ = -q_safe * alpha
    a0_ = 1.0 + alpha
    a1_ = -2.0 * cosw0
    a2_ = 1.0 - alpha

    inv_a0 = 1.0 / a0_
    b0 = b0_ * inv_a0
    b1 = b1_ * inv_a0
    b2 = b2_ * inv_a0
    a1 = a1_ * inv_a0
    a2 = a2_ * inv_a0

    return b0, b1, b2, a1, a2


# -------------------------------------------------------------------------
# Public: init
# -------------------------------------------------------------------------


def xddsp_burst_init(
    sr_hz: float,
    freq_start_hz: float,
    freq_end_hz: float,
    decay_time_sec: float,
    q: float,
    amp: float,
    smoothing_alpha: float = 0.1,
    min_env: float = 1e-6,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """
    Initialize the burst generator state and params.

    Parameters
    ----------
    sr_hz : float
        Sample rate in Hz.
    freq_start_hz : float
        Starting center frequency in Hz.
    freq_end_hz : float
        Ending center frequency in Hz.
    decay_time_sec : float
        Time in seconds for the envelope to decay by 60 dB.
    q : float
        Bandpass Q factor.
    amp : float
        Initial envelope amplitude (env_init).
    smoothing_alpha : float, optional
        Frequency smoothing coefficient in [0, 1]. Higher = more responsive.
    min_env : float, optional
        Minimum envelope level below which we consider output effectively silent.

    Returns
    -------
    state : tuple
        Initial state tuple.
    params : tuple
        Parameter tuple (constant over blocks unless updated).
    """
    # Guard against invalid decay_time_sec
    if decay_time_sec <= 0.0:
        decay_time_sec = 1e-3

    decay_coeff = 10.0 ** (-60.0 / (20.0 * decay_time_sec * sr_hz))

    env_init = amp
    env0 = env_init

    # Initial center frequency is freq_start_hz
    fc0 = freq_start_hz

    b0, b1, b2, a1, a2 = _xddsp_burst_compute_biquad_bandpass_coeffs(fc0, q, sr_hz)

    state_env = env0
    state_z1 = 0.0
    state_z2 = 0.0
    state_b0 = b0
    state_b1 = b1
    state_b2 = b2
    state_a1 = a1
    state_a2 = a2
    state_fc_prev = fc0
    state_decay_coeff = decay_coeff

    state = (
        state_env,
        state_z1,
        state_z2,
        state_b0,
        state_b1,
        state_b2,
        state_a1,
        state_a2,
        state_fc_prev,
        state_decay_coeff,
    )

    params = (
        freq_start_hz,
        freq_end_hz,
        q,
        sr_hz,
        smoothing_alpha,
        min_env,
        env_init,
    )

    return state, params


# -------------------------------------------------------------------------
# Public: update_state (for parameter changes between blocks)
# -------------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def _xddsp_burst_update_state_jit(
    state: Tuple[float, ...],
    params: Tuple[float, ...],
    freq_start_hz: float,
    freq_end_hz: float,
    decay_time_sec: float,
    q: float,
    smoothing_alpha: float,
    min_env: float,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """
    Jitted core for updating state/params when external parameters change.

    All parameters are replaced with the given values.
    """
    sr_hz = params[3]
    env_init = params[6]

    # Guard against invalid decay_time_sec
    if decay_time_sec <= 0.0:
        decay_time_sec_local = 1e-3
    else:
        decay_time_sec_local = decay_time_sec

    decay_coeff = 10.0 ** (-60.0 / (20.0 * decay_time_sec_local * sr_hz))

    # Keep current envelope and filter states, but update decay coeff and
    # biquad coefficients according to the *current* smoothed frequency.
    state_env = state[0]
    state_z1 = state[1]
    state_z2 = state[2]
    state_fc_prev = state[8]

    b0, b1, b2, a1, a2 = _xddsp_burst_compute_biquad_bandpass_coeffs(
        state_fc_prev, q, sr_hz
    )

    state_new = (
        state_env,
        state_z1,
        state_z2,
        b0,
        b1,
        b2,
        a1,
        a2,
        state_fc_prev,
        decay_coeff,
    )

    params_new = (
        freq_start_hz,
        freq_end_hz,
        q,
        sr_hz,
        smoothing_alpha,
        min_env,
        env_init,
    )

    return state_new, params_new


def xddsp_burst_update_state(
    state: Tuple[float, ...],
    params: Tuple[float, ...],
    freq_start_hz: float,
    freq_end_hz: float,
    decay_time_sec: float,
    q: float,
    smoothing_alpha: float,
    min_env: float,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """
    Update burst state and params when external parameters change.

    Parameters
    ----------
    state : tuple
        Current state tuple.
    params : tuple
        Current params tuple.
    freq_start_hz : float
        New starting frequency.
    freq_end_hz : float
        New ending frequency.
    decay_time_sec : float
        New decay time (seconds for -60 dB).
    q : float
        New Q.
    smoothing_alpha : float
        New smoothing alpha.
    min_env : float
        New minimum envelope threshold.

    Returns
    -------
    state_new : tuple
        Updated state.
    params_new : tuple
        Updated params.
    """
    return _xddsp_burst_update_state_jit(
        state,
        params,
        freq_start_hz,
        freq_end_hz,
        decay_time_sec,
        q,
        smoothing_alpha,
        min_env,
    )


# -------------------------------------------------------------------------
# Public: tick
# -------------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def _xddsp_burst_tick_jit(
    x: float,
    state: Tuple[float, ...],
    params: Tuple[float, ...],
) -> Tuple[float, Tuple[float, ...]]:
    """
    Jitted DSP core: process one sample of input x.
    """
    # Unpack state
    env = state[0]
    z1 = state[1]
    z2 = state[2]
    b0 = state[3]
    b1 = state[4]
    b2 = state[5]
    a1 = state[6]
    a2 = state[7]
    fc_prev = state[8]
    decay_coeff = state[9]

    # Unpack params
    freq_start_hz = params[0]
    freq_end_hz = params[1]
    q = params[2]
    sr_hz = params[3]
    smoothing_alpha = params[4]
    min_env = params[5]
    env_init = params[6]

    # Envelope update
    env_new = env * decay_coeff
    if env_new < 0.0:
        env_new = 0.0

    # Early-out effectively silent but keep everything stable
    if env_new < min_env:
        # Envelope is very small; just move envelope towards zero and keep filter state
        state_new = (
            env_new,
            z1,
            z2,
            b0,
            b1,
            b2,
            a1,
            a2,
            fc_prev,
            decay_coeff,
        )
        return 0.0, state_new

    # Normalized envelope for frequency mapping
    eps = 1e-12
    env_norm = env_new / (env_init + eps)

    # Target frequency based on envelope (sweeps from start -> end)
    fc_target = freq_end_hz + (freq_start_hz - freq_end_hz) * env_norm

    # Frequency smoothing
    alpha = smoothing_alpha
    if alpha < 0.0:
        alpha = 0.0
    if alpha > 1.0:
        alpha = 1.0

    fc_smooth = alpha * fc_target + (1.0 - alpha) * fc_prev

    # Recompute biquad coefficients for smoothed frequency
    b0_new, b1_new, b2_new, a1_new, a2_new = _xddsp_burst_compute_biquad_bandpass_coeffs(
        fc_smooth, q, sr_hz
    )

    # Direct Form II Transposed bandpass
    y_bp = b0_new * x + z1
    z1_new = b1_new * x + z2 - a1_new * y_bp
    z2_new = b2_new * x - a2_new * y_bp

    # Apply envelope
    y = y_bp * env_new

    # Pack new state
    state_new = (
        env_new,
        z1_new,
        z2_new,
        b0_new,
        b1_new,
        b2_new,
        a1_new,
        a2_new,
        fc_smooth,
        decay_coeff,
    )

    return y, state_new


def xddsp_burst_tick(
    x: float,
    state: Tuple[float, ...],
    params: Tuple[float, ...],
) -> Tuple[float, Tuple[float, ...]]:
    """
    Public tick wrapper: process one sample.

    Parameters
    ----------
    x : float
        Input sample (e.g., white noise).
    state : tuple
        Current state.
    params : tuple
        Parameter tuple.

    Returns
    -------
    y : float
        Output sample.
    state_new : tuple
        Updated state.
    """
    return _xddsp_burst_tick_jit(x, state, params)


# -------------------------------------------------------------------------
# Public: process (block)
# -------------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def _xddsp_burst_process_jit(
    x: np.ndarray,
    y: np.ndarray,
    state: Tuple[float, ...],
    params: Tuple[float, ...],
) -> Tuple[np.ndarray, Tuple[float, ...]]:
    """
    Jitted core for block processing.

    Notes
    -----
    - `y` must be preallocated with the same shape as `x`.
    - No dynamic allocation occurs inside this function.
    """
    n = x.shape[0]
    state_local = state
    for i in range(n):
        yi, state_local = _xddsp_burst_tick_jit(x[i], state_local, params)
        y[i] = yi
    return y, state_local


def xddsp_burst_process(
    x: np.ndarray,
    state: Tuple[float, ...],
    params: Tuple[float, ...],
) -> Tuple[np.ndarray, Tuple[float, ...]]:
    """
    Process a block of samples.

    Parameters
    ----------
    x : np.ndarray
        Input array of shape (N,), typically white noise.
    state : tuple
        Current state.
    params : tuple
        Parameter tuple.

    Returns
    -------
    y : np.ndarray
        Output array of shape (N,).
    state_new : tuple
        Updated state after processing the block.
    """
    y = np.zeros_like(x)
    y, state_new = _xddsp_burst_process_jit(x, y, state, params)
    return y, state_new


# -------------------------------------------------------------------------
# Smoke test / Example usage
# -------------------------------------------------------------------------


def _example_burst_render():
    """
    Render a short example burst, plot waveform and spectrogram, and optionally play it.
    """
    import matplotlib.pyplot as plt

    try:
        import sounddevice as sd
    except ImportError:
        sd = None

    sr_hz = 48000.0
    duration_sec = 0.5
    n_samples = int(sr_hz * duration_sec)

    # White noise input (pure NumPy; not inside jitted code)
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=n_samples).astype(np.float64)

    # Initialize burst
    freq_start_hz = 8000.0
    freq_end_hz = 500.0
    decay_time_sec = 0.3
    q = 8.0
    amp = 1.0
    smoothing_alpha = 0.2
    min_env = 1e-6

    state, params = xddsp_burst_init(
        sr_hz=sr_hz,
        freq_start_hz=freq_start_hz,
        freq_end_hz=freq_end_hz,
        decay_time_sec=decay_time_sec,
        q=q,
        amp=amp,
        smoothing_alpha=smoothing_alpha,
        min_env=min_env,
    )

    # Process block
    y, state = xddsp_burst_process(x, state, params)

    # Plot waveform
    t = np.arange(n_samples) / sr_hz

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, y)
    plt.title("xddsp_burst - Time Domain")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Spectrogram
    plt.subplot(2, 1, 2)
    plt.specgram(y, NFFT=1024, Fs=sr_hz, noverlap=512, cmap="magma")
    plt.title("Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()

    # Optional listen
    if sd is not None:
        print("Playing burst...")
        sd.play(y * 0.2, int(sr_hz))
        sd.wait()
    else:
        print("sounddevice not installed; skipping playback.")


if __name__ == "__main__":
    _example_burst_render()
```

---

## Things We Can Improve Next

Some natural next steps / variants you might want:

1. **Per-sample parameter control inputs**

   * Allow `freq_start_hz`, `freq_end_hz`, or `q` to be time-varying control arrays rather than scalars.
   * This can be done by extending `xddsp_burst_process` to accept optional control arrays and pass per-sample values into the jitted loop.

2. **Stereo / multichannel version**

   * Create `xddsp_burst_process_stereo` that processes 2 channels in parallel with shared envelope but different filters, or fully independent bursts per channel.

3. **Alternate envelopes**

   * Replace the simple exponential with:

     * Exponential attack + decay (AD),
     * Perceptually linear-in-dB segments,
     * Or multi-stage envelopes (ADSR-like) while keeping state functional.

4. **Nonlinear saturation after bandpass**

   * Add a lightweight nonlinearity (e.g., tanh, soft clipping) after the bandpass to create more “analog” or aggressive bursts.

5. **Trigger-driven burst sequencer module**

   * A companion module that takes a trigger stream and reinitializes burst state (env, fc_prev, etc.) on triggers, allowing classic drum/FX synthesis inside a larger XDDSP patch.

If you’d like, I can spin out any of these (e.g., a triggerable drum-burst variant or a stereo burst processor) in the same XDDSP style.
