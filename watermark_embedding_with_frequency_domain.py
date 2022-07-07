import cv2
import pywt
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


class Arnold:
    def __init__(self, a: int, b: int, rounds: int):
        self.__a = a
        self.__b = b
        self.__rounds = rounds

    def mapping(self, s: np.shape):
        x, y = np.meshgrid(range(s[0]), range(s[0]), indexing="ij")
        xmap = (self.__a * self.__b * x + x + self.__a * y) % s[0]
        ymap = (self.__b * x + y) % s[0]
        return xmap, ymap

    def inverseMapping(self, s: np.shape):
        x, y = np.meshgrid(range(s[0]), range(s[0]), indexing="ij")
        xmap = (x - self.__a * y) % s[0]
        ymap = (-self.__b * x + self.__a * self.__b * y + y) % s[0]
        return xmap, ymap

    def applyTransformTo(self, image: np.ndarray):
        xm, ym = self.mapping(image.shape)
        img = image
        for r in range(self.__rounds):
            img = img[xm, ym]
        return img

    def applyInverseTransformTo(self, image: np.ndarray):
        xm, ym = self.inverseMapping(image.shape)
        img = image
        for r in range(self.__rounds):
            img = img[xm, ym]
        return img


def dct_block(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct_block(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


if __name__ == '__main__':
    main_image_path = "banded-gecko-nicolas-reusens.jpg"
    main_image = cv2.imread(main_image_path)

    watermark_image_path = "logo_image.png"
    watermark_image = cv2.imread(watermark_image_path)

    scale_percent = 10
    scale_width = int(watermark_image.shape[1] * scale_percent / 100)
    scale_height = int(watermark_image.shape[0] * scale_percent / 100)
    watermark_image = cv2.resize(watermark_image, (scale_width, scale_height))

    # DCT of image
    main_image_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    main_image_dct = dct_block(main_image_gray.astype(float))

    # Haar wavelet
    coeffs2 = pywt.dwt2(main_image_dct, 'haar')

    # Haar wavelet coeffs
    LL, (LH, HL, HH) = coeffs2

    # Arnold function transforms
    a = 1
    b = 0
    rounds = 1
    arnold = Arnold(a, b, rounds)

    watermark_image_scrambled = arnold.applyTransformTo(watermark_image)
    watermark_image_scrambled_fromArray = Image.fromarray(watermark_image_scrambled).convert("L")
    watermark_image_scrambled_fromArray.show()

    # DCT to image with Arnold transforms
    watermark_image_scrambled_dct = dct_block(watermark_image_scrambled.astype(float))

    # Embedding of received watermark blocks to Haar wavelet coeffs
    k = 20  # scaling value

    # List of coefficients from watermark into 4 equal parts
    watermark_image_scrambled_dct_blocks = np.split(watermark_image_scrambled_dct, 2)
    watermark_image_scrambled_dct_blocks = [np.hsplit(i, 2) for i in watermark_image_scrambled_dct_blocks]

    wavelet_y, wavelet_x = LL.shape
    scrambled_size = len(watermark_image_scrambled_dct_blocks[0][0])

    embed_counter_x = wavelet_x if wavelet_x < scrambled_size else scrambled_size
    embed_counter_y = wavelet_y if wavelet_y < scrambled_size else scrambled_size

    for i in range(embed_counter_x):
        for j in range(embed_counter_y):
            LL[i][j] += watermark_image_scrambled_dct_blocks[0][0][i][j][0] * k
            LH[i][j] += watermark_image_scrambled_dct_blocks[0][1][i][j][0] * k
            HL[i][j] += watermark_image_scrambled_dct_blocks[1][0][i][j][0] * k
            HH[i][j] += watermark_image_scrambled_dct_blocks[1][1][i][j][0] * k

    coeffs = (LL, (LH, HL, HH))
    main_image_dct_reversedWavelet = pywt.idwt2(coeffs, 'haar')
    main_image_dct_reversedWavelet_idct = idct_block(main_image_dct_reversedWavelet)

    # Round coeffs
    print(main_image_dct_reversedWavelet_idct.shape)
    main_image_dct_reversedWavelet_idct_x = main_image_dct_reversedWavelet_idct.shape[1]
    main_image_dct_reversedWavelet_idct_y = main_image_dct_reversedWavelet_idct.shape[0]

    for i in range(main_image_dct_reversedWavelet_idct_y):
        for j in range(main_image_dct_reversedWavelet_idct_x):
            main_image_dct_reversedWavelet_idct[i][j] = round(main_image_dct_reversedWavelet_idct[i][j], 1)

    main_image_dct_reversedWavelet_idct_fromArray = Image.fromarray(main_image_dct_reversedWavelet_idct).convert("L")
    main_image_dct_reversedWavelet_idct_fromArray.show()
