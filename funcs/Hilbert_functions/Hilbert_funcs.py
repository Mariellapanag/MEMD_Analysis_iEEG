import math
from scipy import angle, unwrap
from scipy.signal import hilbert
import numpy as np


def inst_features(signal, fs ):
    """
    Compute the instantaneous frequency, amplitude and phase for a univariate signal through time T = length of the signal.
    First, Hilbert transform is applied to the signal to extract the analytical form of the signal and then
    we obtain the instantaneous frequency, amplitude and phase of the signal as a function of time.

    Parameters
    ----------
    signal : numpy array
        Array depicts a signal sampled with specified frequency
    fs : int
        Number of generated samples.
        Specify this parameter based on the sampling frequency of the initial signal.
        It might be useful to visualise the data using different fs, e.g. if sampling frequency
        of the initial signal is 1/30 s then if we want to display the results daily, we should
        set this parameter as fs = 2*60*24 (as 1 point of our data corresponds to 30s).

    Returns
    -------
    instantaneous_frequency : numpy array
        Instantaneous frequency using fs defined by the user.
    amplitude_envelope : numpy array
        Instantaneous amplitude using fs defined by the user.
    instantaneous_phase : numpy array
        Instantaneous unwrapped phase using fs defined by the user.
    phase_wrap : numpy array
        Instantaneous wrapped phase using fs defined by the user.
    phase_angle : numpy array
        Instantaneous angle phase using fs defined by the user.
        np.angle ( analytical_signal )

    References
    ----------
    1. `Example from Scipy where the Hilbert transform is applied to determine the amplitude envelope and instantaneous frequency of an amplitude-modulated signal <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html>`_.
    """

    # Compute the hilbert transform
    analytical_signal = hilbert ( signal )

    # Instantaneous frequency
    # instantaneous_frequency = (np.diff(instantaneous_phase) /(2.0*np.pi)* fs)
    instantaneous_frequency = fs / (2 * math.pi) * np.diff ( unwrap ( angle ( analytical_signal ) ) )
    amplitude_envelope = np.abs ( analytical_signal )[1:]
    instantaneous_phase = np.unwrap ( np.angle ( analytical_signal ) )[1:]
    phase_wrap = instantaneous_phase % (2*np.pi)
    phase_angle = np.angle ( analytical_signal )[1:]

    return instantaneous_frequency, amplitude_envelope, instantaneous_phase, phase_wrap, phase_angle
