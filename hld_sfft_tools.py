import numpy as np

from scipy.signal import get_window

def polypeak(signal, fit_point=3, low_f_skip=0):
    """Finds the largest value in a data set by fitting a parabola.

    It picks the largest point in the dataset and fits a quadratic parabola 
    to it. It then uses that to get the interpolated maximum.

    Parameters
    ----------
    signal : numpy.ndarray
        The data to interpolate the highest value of.
    fit_point : int, optional
        The number of points to use in the fit. Needs to be odd.
    low_f_skip : int, optional
        The number of points to disregard at the start of the data.

    Returns
    -------
    x : float
        The x position as a rational number in relation to the index of the
        maximal value.
    y : float
        The y position of the interpolated maximum value.
    """

    if fit_point%2 != 1:
        ValueError(
            f"fit_point needs to be odd, was {fit_point} .")
    m = int((fit_point - 1)/2)  # Number of points either side
    i = low_f_skip + np.argmax(signal[low_f_skip:])  # Find highest point
    # Fit a quadratic and get the x and y of the highest point
    a, b, c = np.polyfit(np.arange(i-m,i+m+1), signal[i-m:i+m+1], 2) 
    x = -0.5*b/a  
    y = a*x**2 + b*x + c
    return x, y


def scanning_fft(signal, fs, tseg, tstep, 
        nfft_power=4, window='hamming', fit_point=5, low_f_skip=100,
        tqdm_bar=None):
    """Finds the changing dominate frequency of a oscillatory signal.

    Finds how the frequency of a oscillatory signal changes with time. This 
    is achieved by performing many Fast Fourier Transforms (FFT) over a 
    small window of signal which is slid along the complete signal. This is 
    useful for extracting the measurement from PDO and TDO experiments.

    Parameters
    ----------
    signal : numpy.ndarray
        The data to extract the signal from in the form of a 1d array.
    fs : float
        The sample frequency of the measurement signal in Hertz.
    tseg : float
        The length in time to examine for each FFT in seconds.
    tstep : float
        How far to shift the window between each FFT in seconds.
    nfft_power : int, optional
        Increases the number of points used in the FFT. As power of two are 
        the most efficient in FFTs it doubles the data by this number.
    window : str, optional
        The windowing function to used for the FFT that will be passed to 
        :func:`scipy.signal.get_window`. THe default is 'hamming'.
    fit_points : int, optional
        The number of points to fit a parabola to identify the peak of the 
        FFT. THe default is `5` and this is passed to :func:`polypeak`.
    low_f_slip : int, optional
        The number of points to skip when identifying the peak at the 
        beginning of the FFT to ignore the low freq upturn. The default is 
        `100` and this is passed to :func:`polypeak`.
    tqdm_bar : `tqdm.tqdm`, optional
        This function can be slow so a tqdm progress bar can be passed using 
        this keyword which will be updated to show the progress of the 
        calculation. This is done by::

            from tqdm import tqdm
            with tqdm() as bar: 
                res = scanning_fft(signal, fs, tseg, tstep, tqdm_bar=bar)

    Returns
    -------
    times : numpy.ndarray
        The midpoint of the time windows which the FFTs where taken at in 
        seconds.
    freqs : numpy.ndarray
        The frequencies of the dominate oscillatory signal against time in 
        Hertz.
    amps : numpy.ndarray
        The amplitude of the oscillatory signal from the FFT. This should be 
        in the units of the signal.
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError(
            f"signal needs to be a numpy array but was instead a "
            f"{type(signal)}")

    if not isinstance(tseg, (int, float, np.int_, np.float_)):
        raise TypeError(
            f"tseg is of type {type(tseg)} but needs to be an float.")

    if not isinstance(tstep, (int, float, np.int_, np.float_)):
        raise TypeError(
            f"tstep is of type {type(tstep)} but needs to be an float.")

    if not isinstance(nfft_power, (int, np.int_)):
        raise TypeError(
            f"nfft_power is of type {type(nfft_power)} but needs to be an "
            "integer.")

    nperseg = int(tseg*fs)
    nstep = int(tstep*fs)
    nfft = 2**(int(np.log2(nperseg))+1+nfft_power)

    if nperseg > len(signal) or nstep > len(signal):
        raise ValueError(
            f"The segments lengths are longer then the total length of the "
            f"data which is probably a mistake.")

    ntimepoint = int((len(signal)-nperseg+nstep)/nstep)
    idx_times = int(nperseg/2)+np.arange(ntimepoint)*nstep
    freqs_i = np.zeros(ntimepoint)
    amps = np.zeros(ntimepoint)

    window = get_window(window, nperseg)

    if tqdm_bar is not None:
        tqdm_bar.reset(total=ntimepoint)

    for x in range(ntimepoint):
        if tqdm_bar is not None:
            tqdm_bar.update()
        freqs_i[x], amps[x] = polypeak(
            np.abs(
                np.fft.rfft(signal[x*nstep:nperseg+x*nstep]*window, n=nfft)),
            fit_point=fit_point, low_f_skip=low_f_skip)

    freqs = freqs_i*fs/nfft
    amps = 2*amps/np.average(window)/nperseg

    return idx_times, freqs, amps


def PUtoB(PU_signal, field_factor, fit_points):
    """Converts the voltage from the pick up coil to field.

    This is used for pulsed field measurements, where to obtain the field
    the induced voltage in a coil is integrated. A fit is also applied 
    because slight differences in the grounding voltage can cause a large 
    change in the field so this needs to be corrected for.

    Parameters
    ----------
    PU_signal : numpy.ndarray
        The signal from pick up coil.
    field_factor : float
        Factor to convert integral to magnetic field. Bare in mind this will 
        change if the acquisition rate changes, for the same coil.
    fit_points : int
        Number of point at each end to remove offset.
    
    Returns
    -------
    field : numpy.ndarray, Data
        An array of magnetic field the same length as PU_signal.
    """
    # Fit the background
    count = np.arange(len(PU_signal))
    ends = np.concatenate([count[:fit_points], count[-fit_points:]])
    a, b = np.polyfit(ends, PU_signal[ends], 1)
    # Calculate magnetic field
    PU_flat = PU_signal - a*count - b
    field_values = np.cumsum(PU_flat*field_factor)

    return field_values
