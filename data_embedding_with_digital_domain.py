from PIL import Image
from tkinter import filedialog
from math import log10, sqrt
import numpy as np
from matplotlib import pyplot as plt
import cv2

# QIM algorithm value
Q = 14

# Stop-symbol for image
END_MESSAGE_FLAG = '1010101010101010101010101010101010101010'


def image_to_bit(image):
    return ''.join([''.join(["{0:08b}".format(i) for i in j]) for j in image.getdata()])


def image_from_bit(message):
    image_pixels = [message[0 + 24 * i:24 * (i + 1)] for i in range(len(message) // 24)]

    result = [[int(image_pixels[i][0:8], 2),
               int(image_pixels[i][8:16], 2),
               int(image_pixels[i][16:25], 2)] for i in range(len(image_pixels))]

    return result


def pixel_encode(p, q, b):
    return q * int(p / q) + int(q / 2) * int(b)


def pixel_decode_c0(p, q):
    return q * int(p / q)


def pixel_decode_c1(p, q):
    return q * int(p / q) + int(q / 2)


def get_image_path():
    return filedialog.askopenfilename(title='Choose image')


def encode(image, message):
    message_len = len(message)
    message_counter = 0

    for row in image:
        for i in range(len(row)):
            if message_counter < message_len:
                row[i] = pixel_encode(row[i], Q, message[message_counter])

                if row[i] < 0:
                    row[i] = 0

                if row[i] > 255:
                    row[i] = 255

                message_counter += 1

    return image


def decode(image):
    result = ''

    for row in image:
        for i in range(len(row)):
            if abs(row[i] - pixel_decode_c0(row[i], Q)) < abs(row[i] - pixel_decode_c1(row[i], Q)):
                result += '0'
            else:
                result += '1'

            if result[-len(END_MESSAGE_FLAG):] == END_MESSAGE_FLAG:
                return result[:-len(END_MESSAGE_FLAG)]

    print('You have no secret message')


def measure(image_original, image_compressed):
    mse = np.mean((np.array(image_original) - np.array(image_compressed)) ** 2)
    rmse = sqrt(mse)
    max_pixel = 255
    psnr = 20 * log10(max_pixel / sqrt(mse))

    print(f'MSE is {mse}')
    print(f'RMSE is {rmse}')
    print(f'PSNR is {psnr} dB')


def image_hist_make(image_path_1, image_path_2):
    image_1 = cv2.imread(image_path_1)

    colors = ("b", "g", "r")

    plt.figure(1)
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image_1], [i], None, [256], [0, 256])
        plt.title("Image 1")
        plt.plot(hist, color=color)
        plt.xlim([0, 260])

    image_2 = cv2.imread(image_path_2)

    plt.figure(2)
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image_2], [i], None, [256], [0, 256])
        plt.title("Image 2")
        plt.plot(hist, color=color)
        plt.xlim([0, 260])

    plt.show()


if __name__ == '__main__':
    while True:
        print('Input "1" to encode image ~')
        print('Input "2" to decode image ~')
        print('Input "3" to analysis images ~')
        print('Input "Q" to exit ~')

        user_input = input('Choose: ')
        if user_input == '1':
            file_path_1 = get_image_path()
            file_path_2 = get_image_path()

            print("image_to_encode")
            image_to_encode = Image.open(file_path_1).convert('RGB')
            image_size = image_to_encode.size
            image_message = Image.open(file_path_2).convert('RGB')

            image_1_list = [list(i) for i in image_to_encode.getdata()]

            print("message")
            message = image_to_bit(image_message) + END_MESSAGE_FLAG

            print("image_tuple")
            image_tuple = [tuple(i) for i in encode(image_1_list, message)]

            print("image_encoded")
            image_encoded = Image.new(mode='RGB', size=image_size)
            print("image_encoded.putdata")
            image_encoded.putdata(data=image_tuple)
            print("image_encoded.save")
            image_encoded.save('encoded.png')

        elif user_input == '2':
            file_path_1 = get_image_path()
            image_to_decode = Image.open(file_path_1).convert('RGB')

            image_1_list = [list(i) for i in image_to_decode.getdata()]

            message = image_from_bit(decode(image_1_list))
            image_tuple = [tuple(i) for i in message]
            image_decoded = Image.new(mode='RGB', size=(16, 16))
            image_decoded.putdata(data=image_tuple)
            image_decoded.save('decoded.png')

        elif user_input == '3':
            file_path_1 = get_image_path()
            file_path_2 = get_image_path()

            image_1 = Image.open(file_path_1).convert('RGB')
            image_1_list = [list(i) for i in image_1.getdata()]

            image_2 = Image.open(file_path_2).convert('RGB')
            image_2_list = [list(i) for i in image_2.getdata()]

            measure(image_1_list, image_2_list)
            image_hist_make(file_path_1, file_path_2)

        elif user_input == 'Q':
            break

        else:
            print('Unknown command.')
