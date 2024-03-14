import pywt
import pywt.data
import numpy as np
from PIL import Image
from scipy.ndimage import filters
import cv2

return_lst = []
fingerprint = []

def denoise(image_set):
    for image in image_set:
        image = check_orientation(image)
        print(f"image size: {image.size}")
        coeffs = pywt.dwt2(image, 'db4')
        LL, (LH, HL, HH) = coeffs
        LH = np.array(LH)
        # print(LH)
        HL = np.array(HL)
        # print(HL)
        HH = np.array(HH)
        # print(HH)
        image_final = np.add(LH, HL, HH)
        return_lst.append(image_final)
    return average_fingerprint(return_lst)

def average_fingerprint(fingerprint_list):
    hold = np.array(fingerprint_list[0])
    for img_rep in fingerprint_list[1:]:
        x = np.array(img_rep)
        hold = np.add(hold,x)
    average = np.divide(hold, len(return_lst))

    print("Average: ")
    checksum = 0
    for i in average:
        checksum += sum(i)
    print(checksum)
    print(sum(average))
    returned_image = Image.fromarray(average, mode='L')
    fingerprint.append(returned_image)
    #return fingerprint

def main(image_set):
    return_lst.clear()
    fingerprint.clear()

    for image in image_set:
        print(f"image size: {image.size}")
        img = np.asarray(image)
        return_lst.append(noise_extract(img[:800, :1000]))

    average_fingerprint(return_lst)
    return fingerprint

def noise_extract(im: np.ndarray, levels: int = 4, sigma: float = 5) -> np.ndarray:
    """
    NoiseExtract as from Binghamton toolbox.

    :param im: grayscale or color image, np.uint8
    :param levels: number of wavelet decomposition levels
    :param sigma: estimated noise power
    :return: noise residual
    """

    assert (im.dtype == np.uint8)
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
                wlet = pywt.wavedec2(im[:, :, ch], 'db4', level=levels)
            except ValueError:
                levels -= 1
                wlet = None
        if wlet is None:
            raise ValueError('Impossible to compute Wavelet filtering for input size: {}'.format(im.shape))

        wlet_details = wlet[1:]

        wlet_details_filter = [None] * len(wlet_details)
        # Cycle over Wavelet levels 1:levels-1
        for wlet_level_idx, wlet_level in enumerate(wlet_details):
            # Cycle over H,V,D components
            level_coeff_filt = [None] * 3
            for wlet_coeff_idx, wlet_coeff in enumerate(wlet_level):
                level_coeff_filt[wlet_coeff_idx] = wiener_adaptive(wlet_coeff, noise_var)
            wlet_details_filter[wlet_level_idx] = tuple(level_coeff_filt)

        # Set filtered detail coefficients for Levels > 0 ---
        wlet[1:] = wlet_details_filter

        # Set to 0 all Level 0 approximation coefficients ---
        wlet[0][...] = 0

        # Invert wavelet transform ---
        wrec = pywt.waverec2(wlet, 'db4')
        try:
            W[:, :, ch] = wrec
        except ValueError:
            W = np.zeros(wrec.shape[:2] + (im.shape[2],), np.float32)
            W[:, :, ch] = wrec

    if W.shape[2] == 1:
        W.shape = W.shape[:2]

    W = W[:im.shape[0], :im.shape[1]]

    print(type(W))

    return W

def wiener_adaptive(x: np.ndarray, noise_var: float, **kwargs) -> np.ndarray:
    """
    WaveNoise as from Binghamton toolbox.
    Wiener adaptive flter aimed at extracting the noise component
    For each input pixel the average variance over a neighborhoods of different window sizes is first computed.
    The smaller average variance is taken into account when filtering according to Wiener.
    :param x: 2D matrix
    :param noise_var: Power spectral density of the noise we wish to extract (S)
    :param window_size_list: list of window sizes
    :return: wiener filtered version of input x
    """
    window_size_list = list(kwargs.pop('window_size_list', [3, 5, 7, 9]))

    energy = x ** 2

    avg_win_energy = np.zeros(x.shape + (len(window_size_list),))
    for window_idx, window_size in enumerate(window_size_list):
        avg_win_energy[:, :, window_idx] = filters.uniform_filter(energy,
                                                                  window_size,
                                                                  mode='constant')

    coef_var = threshold(avg_win_energy, noise_var)
    coef_var_min = np.min(coef_var, axis=2)

    x = x * noise_var / (coef_var_min + noise_var)

    return x


def threshold(wlet_coeff_energy_avg: np.ndarray, noise_var: float) -> np.ndarray:
    """
    Noise variance theshold as from Binghamton toolbox.
    :param wlet_coeff_energy_avg:
    :param noise_var:
    :return: noise variance threshold
    """
    res = wlet_coeff_energy_avg - noise_var
    return (res + np.abs(res)) / 2

def check_orientation(image):
    width, height = image.size
    if width < height:
        print(f"Original Size: {image.size}")
        rotated_image = image.rotate(90, expand=True) # This angle is based on the assumption most cameras have the shutter button on the right so teh camera is usually rotated 90 degrees counter clockwise to take portrait photos.
        print(f"Rotated size: {rotated_image.size}")
        return rotated_image
    elif width > height:
        print("Image Already Portrait")
        return image
    else:
        return image # This creates an issue as we cannot confirm the orientation the image should be in using this method.