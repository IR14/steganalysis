import numpy as np
from PIL import Image
from argparse import ArgumentParser
import jpegio as jio

BYTE = 8


def image_to_bits(pixels):
    bits = ''

    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            for color in range(3):
                bits += bin(pixels[i, j, color])[2:].rjust(BYTE, '0')

    return bits


def bits_to_image(bits):
    assert len(bits) % BYTE == 0, 'Wrong length of bits'

    idx = 0

    sqrt = int((len(bits) // (BYTE * 3)) ** 0.5)

    pixels = np.zeros(shape=(sqrt, sqrt, 4), dtype=np.uint8) + 255  # +255 for alpha channel

    for i in range(sqrt):
        for j in range(sqrt):
            for color in range(3):
                pixels[i, j, color] = int(bits[idx:idx + 8], 2)
                idx += 8

    return pixels


def encode_with_jsteg(container_path: str, watermark_path: str, output_path: str):
    watermark_bits = image_to_bits(np.array(Image.open(watermark_path)))
    inserted_bits = 0
    skipped = 0
    container = jio.read(container_path)
    done = False

    for color in range(3):
        for i in range(0, container.coef_arrays[color].shape[0], BYTE):
            for j in range(0, container.coef_arrays[color].shape[1], BYTE):
                val = container.coef_arrays[color][i, j]

                if val in [-1, 0, 1]:
                    skipped += 1
                    continue

                if watermark_bits[inserted_bits] == '0':
                    # set least significant bit to 0
                    container.coef_arrays[color][i, j] = np.sign(val) * (abs(val) - abs(val) % 2)
                else:
                    # set least significant bit to 1
                    container.coef_arrays[color][i, j] = np.sign(val) * (abs(val) | 1)

                inserted_bits += 1
                if inserted_bits == len(watermark_bits):
                    done = True
                    break

            if done:
                break

        if done:
            break

    if inserted_bits < len(watermark_bits):
        print(f'Inserted only {inserted_bits}/{len(watermark_bits)}, skipped={skipped}.')
        raise RuntimeError('Inserted bits missing')

    print(f'Inserted {inserted_bits} bits.')

    jio.write(container, output_path)
    return inserted_bits


def decode_with_jsteg(container_path: str, watermark_bit_len: int, output_path: str):
    container = jio.read(container_path)
    watermark_bits = ''
    extracted_bits = 0
    done = False
    skipped = 0

    for color in range(3):
        for i in range(0, container.coef_arrays[color].shape[0], BYTE):
            for j in range(0, container.coef_arrays[color].shape[1], BYTE):
                val = container.coef_arrays[color][i, j]

                if val in [-1, 0, 1]:
                    skipped += 1
                    continue

                watermark_bits += str(val & 1)

                extracted_bits += 1
                if extracted_bits == watermark_bit_len:
                    done = True
                    break

            if done:
                break

        if done:
            break

    if extracted_bits < watermark_bit_len:
        print(f'Extracted only {extracted_bits}/{watermark_bit_len} bits.')
        # raise RuntimeError('Extracted bits missing')

    print(f'Extracted {extracted_bits} bits.')

    watermark_pixels = bits_to_image(watermark_bits)
    result_jpg = Image.fromarray(watermark_pixels).convert('RGB')
    result_jpg.save(output_path)
    # Image.fromarray(watermark_pixels).save(output_path)


def encode_with_f3(container_path: str, watermark_path: str, output_path: str):
    watermark_bits = image_to_bits(np.array(Image.open(watermark_path)))
    inserted_bits = 0
    container = jio.read(container_path)
    done = False

    for color in range(3):
        for i in range(0, container.coef_arrays[color].shape[0], BYTE):
            for j in range(0, container.coef_arrays[color].shape[1], BYTE):
                val = container.coef_arrays[color][i, j]

                if val == 0 or (i == 0 and j == 0):
                    continue

                if (abs(val) % 2 != 0 and watermark_bits[inserted_bits] == '0'
                        or abs(val) % 2 == 0 and watermark_bits[inserted_bits] == '1'):
                    # c = sign(c) * (|c| - 1)
                    container.coef_arrays[color][i, j] = np.sign(val) * (abs(val) - 1)
                    if container.coef_arrays[color][i, j] == 0:
                        continue
                else:
                    pass

                inserted_bits += 1
                if inserted_bits == len(watermark_bits):
                    done = True
                    break

            if done:
                break

        if done:
            break

    if inserted_bits < len(watermark_bits):
        print(f'Inserted only {inserted_bits}/{len(watermark_bits)}.')
        raise RuntimeError('Inserted bits missing')

    print(f'Inserted {inserted_bits} bits.')

    jio.write(container, output_path)
    return inserted_bits


def decode_with_f3(container_path: str, watermark_bit_len: int, output_path: str):
    container = jio.read(container_path)
    watermark_bits = ''
    extracted_bits = 0
    done = False

    for color in range(3):
        for i in range(0, container.coef_arrays[color].shape[0], BYTE):
            for j in range(0, container.coef_arrays[color].shape[1], BYTE):
                val = container.coef_arrays[color][i, j]

                if val == 0 or (i == 0 and j == 0):
                    continue

                watermark_bits += str(abs(val) % 2)
                extracted_bits += 1

                if extracted_bits == watermark_bit_len:
                    done = True
                    break

            if done:
                break

        if done:
            break

    if extracted_bits < watermark_bit_len:
        print(f'Extracted only {extracted_bits}/{watermark_bit_len} bits.')
        # raise RuntimeError('Extracted bits missing')

    print(f'Extracted {extracted_bits} bits.')

    watermark_pixels = bits_to_image(watermark_bits)
    result_jpg = Image.fromarray(watermark_pixels).convert('RGB')
    result_jpg.save(output_path)
    # Image.fromarray(watermark_pixels).save(output_path)


def encode_with_f4(container_path: str, watermark_path: str, output_path: str):
    watermark_bits = image_to_bits(np.array(Image.open(watermark_path)))
    inserted_bits = 0
    container = jio.read(container_path)
    done = False

    for color in range(3):
        for i in range(0, container.coef_arrays[color].shape[0], BYTE):
            for j in range(0, container.coef_arrays[color].shape[1], BYTE):
                val = container.coef_arrays[color][i, j]

                if val == 0 or (i == 0 and j == 0):
                    continue

                if (val < 0 and (abs(val) % 2 != 0 and watermark_bits[inserted_bits] == '1'
                                 or abs(val) % 2 == 0 and watermark_bits[inserted_bits] == '0')):
                    container.coef_arrays[color][i, j] = val + 1

                elif (val > 0 and (abs(val) % 2 != 0 and watermark_bits[inserted_bits] == '0'
                                   or abs(val) % 2 == 0 and watermark_bits[inserted_bits] == '1')):
                    container.coef_arrays[color][i, j] = val - 1

                if container.coef_arrays[color][i, j] == 0:
                    # should use inserted bit again
                    continue

                inserted_bits += 1
                if inserted_bits == len(watermark_bits):
                    done = True
                    break

            if done:
                break

        if done:
            break

    if inserted_bits < len(watermark_bits):
        print(f'Inserted only {inserted_bits}/{len(watermark_bits)}.')
        raise RuntimeError('Inserted bits missing')

    print(f'Inserted {inserted_bits} bits.')

    jio.write(container, output_path)
    return inserted_bits


def decode_with_f4(container_path: str, watermark_bit_len: int, output_path: str):
    container = jio.read(container_path)
    watermark_bits = ''
    extracted_bits = 0
    done = False

    for color in range(3):
        for i in range(0, container.coef_arrays[color].shape[0], BYTE):
            for j in range(0, container.coef_arrays[color].shape[1], BYTE):
                val = container.coef_arrays[color][i, j]

                if val == 0 or (i == 0 and j == 0):
                    continue

                if val > 0 and abs(val) % 2 == 0 or val < 0 and abs(val) % 2 != 0:
                    watermark_bits += '0'
                else:
                    watermark_bits += '1'

                extracted_bits += 1
                if extracted_bits == watermark_bit_len:
                    done = True
                    break

            if done:
                break

        if done:
            break

    if extracted_bits < watermark_bit_len:
        print(f'Extracted only {extracted_bits}/{watermark_bit_len} bits.')
        # raise RuntimeError('Extracted bits missing')

    print(f'Extracted {extracted_bits} bits.')

    watermark_pixels = bits_to_image(watermark_bits)
    result_jpg = Image.fromarray(watermark_pixels).convert('RGB')
    result_jpg.save(output_path)
    # Image.fromarray(watermark_pixels).save(output_path)


def change_last_bit(val: int):
    sign = np.sign(val)
    val = abs(val)
    if val % 2 == 0:
        return sign * (val | 1)
    else:
        return sign * (val - 1)


def encode_with_f5(container_path: str, watermark_path: str, output_path: str):
    watermark_bits = image_to_bits(np.array(Image.open(watermark_path)))
    inserted_bits = 0
    container = jio.read(container_path)
    done = False
    coef_pos_to_change = []  # append pairs (coords) here, when = 3 - change

    for color in range(3):
        for i in range(0, container.coef_arrays[color].shape[0], BYTE):
            # coef_arrays - DCT coefficient arrays in jpegio format
            for j in range(0, container.coef_arrays[color].shape[1], BYTE):
                if i == 0 and j == 0:
                    continue

                if len(coef_pos_to_change) < 3:
                    coef_pos_to_change.append((i, j))
                    continue

                a1, a2, a3 = (
                    container.coef_arrays[color][coef_pos_to_change[0][0], coef_pos_to_change[0][1]] & 1,
                    container.coef_arrays[color][coef_pos_to_change[1][0], coef_pos_to_change[1][1]] & 1,
                    container.coef_arrays[color][coef_pos_to_change[2][0], coef_pos_to_change[2][1]] & 1,
                )

                x1, x2 = int(watermark_bits[inserted_bits]), int(watermark_bits[inserted_bits + 1])

                if x1 == a1 ^ a3 and x2 == a2 ^ a3:
                    pass

                elif x1 != a1 ^ a3 and x2 == a2 ^ a3:
                    container.coef_arrays[color][coef_pos_to_change[0][0], coef_pos_to_change[0][1]] = \
                        change_last_bit(
                            container.coef_arrays[color][coef_pos_to_change[0][0], coef_pos_to_change[0][1]]
                        )
                elif x1 == a1 ^ a3 and x2 != a2 ^ a3:
                    container.coef_arrays[color][coef_pos_to_change[1][0], coef_pos_to_change[1][1]] = \
                        change_last_bit(
                            container.coef_arrays[color][coef_pos_to_change[1][0], coef_pos_to_change[1][1]]
                        )
                else:
                    container.coef_arrays[color][coef_pos_to_change[2][0], coef_pos_to_change[2][1]] = \
                        change_last_bit(
                            container.coef_arrays[color][coef_pos_to_change[2][0], coef_pos_to_change[2][1]]
                        )

                coef_pos_to_change.clear()
                inserted_bits += 2
                if inserted_bits == len(watermark_bits):
                    done = True
                    break

            if done:
                break

        if done:
            break

    if inserted_bits < len(watermark_bits):
        print(f'Inserted only {inserted_bits}/{len(watermark_bits)}.')
        raise RuntimeError('Inserted bits missing')

    print(f'Inserted {inserted_bits} bits.')

    jio.write(container, output_path)
    return inserted_bits


def decode_with_f5(container_path: str, watermark_bit_len: int, output_path: str, verbose=False):
    container = jio.read(container_path)
    watermark_bits = ''
    extracted_bits = 0
    done = False
    coef_pos_to_change = []  # append pairs (coords) here, when = 3 - change

    for color in range(3):
        for i in range(0, container.coef_arrays[color].shape[0], BYTE):
            for j in range(0, container.coef_arrays[color].shape[1], BYTE):
                if i == 0 and j == 0:
                    continue

                if len(coef_pos_to_change) < 3:
                    coef_pos_to_change.append((i, j))
                    continue

                a1, a2, a3 = (
                    container.coef_arrays[color][coef_pos_to_change[0][0], coef_pos_to_change[0][1]] & 1,
                    container.coef_arrays[color][coef_pos_to_change[1][0], coef_pos_to_change[1][1]] & 1,
                    container.coef_arrays[color][coef_pos_to_change[2][0], coef_pos_to_change[2][1]] & 1,
                )

                coef_pos_to_change.clear()
                watermark_bits += str(a1 ^ a3) + str(a2 ^ a3)
                extracted_bits += 2
                if extracted_bits == watermark_bit_len:
                    done = True
                    break

            if done:
                break

        if done:
            break

    if extracted_bits < watermark_bit_len:
        print(f'Extracted only {extracted_bits}/{watermark_bit_len} bits.')
        # raise RuntimeError('Extracted bits missing')

    print(f'Extracted {extracted_bits} bits.')

    watermark_pixels = bits_to_image(watermark_bits)
    result_jpg = Image.fromarray(watermark_pixels).convert('RGB')
    result_jpg.save(output_path)
    # Image.fromarray(watermark_pixels).save(output_path)


if __name__ == '__main__':
    aparser = ArgumentParser(description='Options')

    aparser.add_argument(
        '-m', '--mode',
        required=True,
        dest='mode',
        choices=['encode', 'decode'],
        type=str, help='encode or decode command'
    )

    aparser.add_argument(
        '-t', '--type',
        required=True,
        dest='type',
        choices=['jsteg', 'f3', 'f4', 'f5'],
        type=str, help='type of algorithm to use in program'
    )

    aparser.add_argument(
        '-c', '--container',
        dest='container_path',
        required=True,
        type=str, help='path to container-image'
    )

    aparser.add_argument(
        '-w', '--watermark',
        dest='watermark_path',
        default=True,
        type=str, help='path to watermark-image'
    )

    aparser.add_argument(
        '-e', '--output-e',
        dest='output_image',
        default='encoded.jpg',
        type=str, help='path to store encoded image'
    )

    aparser.add_argument(
        '-d', '--output-d',
        dest='output_image_decoded',
        default='decoded.jpg',
        type=str, help='path to store decoded image (watermark)'
    )

    aparser.add_argument(
        '-b', '--bits',
        dest='bit_len',
        default=True,
        type=int, help='bit length of watermark to decode'
    )

    args = aparser.parse_args()
    if args.mode == 'encode':
        if args.type == 'jsteg':
            print('jsteg method:')
            encode_with_jsteg(args.container_path, args.watermark_path, args.output_image)

        if args.type == 'f3':
            print('f3 method:')
            encode_with_f3(args.container_path, args.watermark_path, args.output_image)

        if args.type == 'f4':
            print('f4 method:')
            encode_with_f4(args.container_path, args.watermark_path, args.output_image)

        if args.type == 'f5':
            print('f5 method:')
            encode_with_f5(args.container_path, args.watermark_path, args.output_image)
    else:
        if args.type == 'jsteg':
            print('jsteg method:')
            decode_with_jsteg(args.container_path, args.bit_len, args.output_image_decoded)

        if args.type == 'f3':
            print('f3 method:')
            decode_with_f3(args.container_path, args.bit_len, args.output_image_decoded)

        if args.type == 'f4':
            print('f4 method:')
            decode_with_f4(args.container_path, args.bit_len, args.output_image_decoded)

        if args.type == 'f5':
            print('f5 method:')
            decode_with_f5(args.container_path, args.bit_len, args.output_image_decoded)
