"""
Adapted from <https://github.com/polimi-ispl/prnu-python>:
    @author: Luca Bondi (luca.bondi@polimi.it)
    @author: Paolo Bestagini (paolo.bestagini@polimi.it)
    @author: Nicolò Bonettini (nicolo.bonettini@polimi.it)
    Politecnico di Milano 2018
"""

import numpy as np
import pywt
from numpy.fft import fft2, ifft2
from scipy.ndimage import filters


def extract_prnu_lp(im,
                    levels=int(4),
                    sigma=float(5),
                    wdft_sigma=float(0)):
    """
    Extract noise residual from a single image
    :param im: grayscale or color image, np.uint8
    :param levels: number of wavelet decomposition levels
    :param sigma: estimated noise power
    :param wdft_sigma: estimated DFT noise power
    :return: prnu_lp (with linear pattern)
    """
    prnu_lp = np.zeros_like(im, dtype="float32")
    chans = im.shape[-1]

    V = noise_extract(im, levels, sigma)

    for i in range(chans):
        W = np.copy(V[:, :, i])
        W_std = W.std(ddof=1) if wdft_sigma == 0 else wdft_sigma
        prnu_lp[:, :, i] = wiener_dft(W, W_std).astype(np.float32)
    return prnu_lp


def extract_prnu(im,
                 levels=int(4),
                 sigma=float(5),
                 wdft_sigma=float(0)):
    """
    Extract noise residual from a single image
    :param im: grayscale or color image, np.uint8
    :param levels: number of wavelet decomposition levels
    :param sigma: estimated noise power
    :param wdft_sigma: estimated DFT noise power
    :return: prnu (linear pattern removed)
    """
    prnu = np.zeros_like(im, dtype="float32")
    chans = im.shape[-1]

    V = noise_extract(im, levels, sigma)

    for i in range(chans):
        W = zero_mean_total(np.copy(V[:, :, i]))
        W_std = W.std(ddof=1) if wdft_sigma == 0 else wdft_sigma
        prnu[:, :, i] = wiener_dft(W, W_std).astype(np.float32)
    return prnu


def noise_extract(im, levels=int(4), sigma=float(5)):
    """
    NoiseExtract as from Binghamton toolbox.

    :param im: grayscale or color image, np.uint8
    :param levels: number of wavelet decomposition levels
    :param sigma: estimated noise power
    :return: noise residual
    """

    # assert (im.dtype == np.uint8)
    assert (im.ndim in [2, 3])

    im = im.astype(np.float32)

    noise_var = sigma ** 2

    if im.ndim == 2:
        im.shape += (1,)

    W = np.zeros(im.shape, np.float32)

    for ch in range(im.shape[2]):

        wlet = None
        while wlet is None and levels > 0:
            try:
                wlet = pywt.wavedec2(im[:, :, ch], "db4", level=levels)
            except ValueError:
                levels -= 1
                wlet = None
        if wlet is None:
            raise ValueError(
                "Impossible to compute Wavelet filtering for input "
                "size: {}".format(im.shape))

        wlet_details = wlet[1:]

        wlet_details_filter = [None] * len(wlet_details)
        # Cycle over Wavelet levels 1:levels-1
        for wlet_level_idx, wlet_level in enumerate(wlet_details):
            # Cycle over H,V,D components
            level_coeff_filt = [None] * 3
            for wlet_coeff_idx, wlet_coeff in enumerate(wlet_level):
                level_coeff_filt[wlet_coeff_idx] = wiener_adaptive(
                    wlet_coeff, noise_var)
            wlet_details_filter[wlet_level_idx] = tuple(level_coeff_filt)

        # Set filtered detail coefficients for Levels > 0 ---
        wlet[1:] = wlet_details_filter

        # Set to 0 all Level 0 approximation coefficients ---
        wlet[0][...] = 0

        # Invert wavelet transform ---
        wrec = pywt.waverec2(wlet, "db4")
        try:
            W[:, :, ch] = wrec
        except ValueError:
            W = np.zeros(wrec.shape[:2] + (im.shape[2],), np.float32)
            W[:, :, ch] = wrec

    if W.shape[2] == 1:
        W.shape = W.shape[:2]

    W = W[:im.shape[0], :im.shape[1]]

    return W


def wiener_dft(im, sigma):
    """
    Adaptive Wiener filter applied to the 2D FFT of the image
    :param im: multidimensional array
    :param sigma: estimated noise power
    :return: filtered version of input im
    """
    noise_var = sigma ** 2
    h, w = im.shape

    im_noise_fft = fft2(im)
    im_noise_fft_mag = np.abs(im_noise_fft / (h * w) ** .5)

    im_noise_fft_mag_noise = wiener_adaptive(im_noise_fft_mag, noise_var)

    zeros_y, zeros_x = np.nonzero(im_noise_fft_mag == 0)

    im_noise_fft_mag[zeros_y, zeros_x] = 1
    im_noise_fft_mag_noise[zeros_y, zeros_x] = 0

    im_noise_fft_filt = (im_noise_fft * im_noise_fft_mag_noise /
                         im_noise_fft_mag)
    im_noise_filt = np.real(ifft2(im_noise_fft_filt))

    return im_noise_filt.astype(np.float32)


def zero_mean(im):
    """
    ZeroMean called with the "both" argument, as from Binghamton toolbox.
    :param im: multidimensional array
    :return: zero mean version of input im
    """
    # Adapt the shape ---
    if im.ndim == 2:
        im.shape += (1,)

    h, w, ch = im.shape

    # Subtract the 2D mean from each color channel ---
    ch_mean = im.mean(axis=0).mean(axis=0)
    ch_mean.shape = (1, 1, ch)
    i_zm = im - ch_mean

    # Compute the 1D mean along each row and each column, then subtract ---
    row_mean = i_zm.mean(axis=1)
    col_mean = i_zm.mean(axis=0)

    row_mean.shape = (h, 1, ch)
    col_mean.shape = (1, w, ch)

    i_zm_r = i_zm - row_mean
    i_zm_rc = i_zm_r - col_mean

    # Restore the shape ---
    if im.shape[2] == 1:
        i_zm_rc.shape = im.shape[:2]

    return i_zm_rc


def zero_mean_total(im):
    """
    ZeroMeanTotal as from Binghamton toolbox.
    :param im: multidimensional array
    :return: zero mean version of input im
    """
    im[0::2, 0::2] = zero_mean(im[0::2, 0::2])
    im[1::2, 0::2] = zero_mean(im[1::2, 0::2])
    im[0::2, 1::2] = zero_mean(im[0::2, 1::2])
    im[1::2, 1::2] = zero_mean(im[1::2, 1::2])
    return im


def rgb2gray(im):
    """
    RGB to gray as from Binghamton toolbox.
    :param im: multidimensional array
    :return: grayscale version of input im
    """
    rgb2gray_vector = np.asarray(
        [0.29893602, 0.58704307, 0.11402090]).astype(np.float32)
    rgb2gray_vector.shape = (3, 1)

    if im.ndim == 2:
        im_gray = np.copy(im)
    elif im.shape[2] == 1:
        im_gray = np.copy(im[:, :, 0])
    elif im.shape[2] == 3:
        w, h = im.shape[:2]
        im = np.reshape(im, (w * h, 3))
        im_gray = np.dot(im, rgb2gray_vector)
        im_gray.shape = (w, h)
    else:
        raise ValueError("Input image must have 1 or 3 channels")

    return im_gray.astype(np.float32)


def threshold(wlet_coeff_energy_avg, noise_var):
    """
    Noise variance theshold as from Binghamton toolbox.
    :param wlet_coeff_energy_avg:
    :param noise_var:
    :return: noise variance threshold
    """
    res = wlet_coeff_energy_avg - noise_var
    return (res + np.abs(res)) / 2


def wiener_adaptive(x, noise_var, **kwargs):
    """
    WaveNoise as from Binghamton toolbox.
    Wiener adaptive flter aimed at extracting the noise component
    For each input pixel the average variance over a neighborhoods of
    different window sizes is first computed.
    The smaller average variance is taken into account when filtering
    according to Wiener.
    :param x: 2D matrix
    :param noise_var: Power spectral density of the noise we wish to extract
    (S)
    :param window_size_list: list of window sizes
    :return: wiener filtered version of input x
    """
    window_size_list = list(kwargs.pop("window_size_list", [3, 5, 7, 9]))

    energy = x ** 2

    avg_win_energy = np.zeros(x.shape + (len(window_size_list),))
    for window_idx, window_size in enumerate(window_size_list):
        avg_win_energy[:, :, window_idx] =\
            filters.uniform_filter(energy,
                                   window_size,
                                   mode="constant")

    coef_var = threshold(avg_win_energy, noise_var)
    coef_var_min = np.min(coef_var, axis=2)

    x = x * noise_var / (coef_var_min + noise_var)

    return x
