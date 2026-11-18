import math
import random
import cv2
import numpy as np
from message import Message


def psnr(img1, img2):
    """Compute peak signal-to-noise ratio between two images."""
    rms = math.sqrt(np.sum((img1.astype("float") - img2.astype("float")) ** 2) / (img1.shape[1] * img1.shape[0]))
    return 20 * math.log10(256 / rms)


class BPCS(object):
    """Bit-Plane Complexity Segmentation (BPCS) encoder and decoder helper."""

    def __init__(self, img_path):
        self.img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        self.row, self.col = self.img.shape[0], self.img.shape[1]

    def generate_seed(self, key):
        """Generate deterministic integer seed from a string key."""
        return sum([ord(a) for a in key])

    def get_row_col(self, sequence_number):
        row = sequence_number // 8
        col = sequence_number % 8
        return (row, col)

    def put_msg_randomly(self, img_bitplane, msg_bitplane, rand_seq):
        i = 0
        for row in msg_bitplane:
            for cell in row:
                (x, y) = self.get_row_col(rand_seq[i])
                img_bitplane[x, y] = cell
                i += 1
        return img_bitplane

    def get_msg_randomly(self, img_bitplane, rand_seq):
        msg_bitplane = [[0 for _ in range(8)] for _ in range(8)]
        i = 0
        for row in range(len(msg_bitplane)):
            for col in range(len(msg_bitplane[row])):
                (x, y) = self.get_row_col(rand_seq[i])
                msg_bitplane[row][col] = img_bitplane[x, y]
                i += 1
        return np.array(msg_bitplane)

    def hide(self, message, threshold=0.3, randomize=False, key=None):
        """Embed 8x8 message bit-plane blocks into cover blocks meeting complexity >= threshold."""
        windowsize_r = 8
        windowsize_c = 8

        if randomize:
            random.seed(self.generate_seed(key))

        msg_iterator = 0
        while msg_iterator < len(message):
            for row in range(0, self.row - windowsize_r + 1, windowsize_r):
                for col in range(0, self.col - windowsize_c + 1, windowsize_c):
                    temp_block = self.img[row:row + windowsize_r, col:col + windowsize_c]
                    channels_block = cv2.split(temp_block)
                    channels_bitplane = [self.to_bitplane(block) for block in channels_block]
                    for i in range(len(channels_bitplane)):
                        itr_bitplane = 0
                        while itr_bitplane < len(channels_bitplane[i]) and msg_iterator < len(message):
                            if self.calculate_complexity(channels_bitplane[i][itr_bitplane]) >= threshold:
                                if randomize:
                                    rand_sequence = random.sample(range(64), 64)
                                    channels_bitplane[i][itr_bitplane] = self.put_msg_randomly(
                                        channels_bitplane[i][itr_bitplane], message[msg_iterator], rand_sequence
                                    )
                                else:
                                    channels_bitplane[i][itr_bitplane] = message[msg_iterator]
                                msg_iterator += 1
                            itr_bitplane += 1

                    new_channels = [self.bitplane_to_channel(bitplane) for bitplane in channels_bitplane]
                    temp_block = cv2.merge(new_channels)
                    self.img[row:row + windowsize_r, col:col + windowsize_c] = temp_block

                    if msg_iterator >= len(message):
                        break
                if msg_iterator >= len(message):
                    break
            if msg_iterator >= len(message):
                break
        return self.img

    def show(self, threshold=0.3, randomize=False, key=None):
        """Extract 8x8 blocks whose complexity meets threshold."""
        windowsize_r = 8
        windowsize_c = 8
        msg_bitplane = []

        if randomize:
            random.seed(self.generate_seed(key))

        msg_iterator = 0
        for row in range(0, self.row - windowsize_r + 1, windowsize_r):
            for col in range(0, self.col - windowsize_c + 1, windowsize_c):
                temp_block = self.img[row:row + windowsize_r, col:col + windowsize_c]
                channels_block = cv2.split(temp_block)
                channels_bitplane = [self.to_bitplane(block) for block in channels_block]

                i = 0
                for channel_bitplane in channels_bitplane:
                    j = 0
                    for bitplane in channel_bitplane:
                        if self.calculate_complexity(bitplane) >= threshold:
                            if randomize:
                                rand_sequence = random.sample(range(64), 64)
                                msg_bitplane.append(self.get_msg_randomly(bitplane, rand_sequence))
                            else:
                                msg_bitplane.append(bitplane)
                            msg_iterator += 1
                            if msg_iterator < 5:
                                print("get at", row, col, i, j)
                        j += 1
                    i += 1
        return msg_bitplane

    def to_bitplane(self, img):
        result = []
        for i in reversed(range(8)):
            result.append((img / (2 ** i)).astype(int) % 2)
        return result

    def bitplane_to_channel(self, bitplane):
        result = (
            2
            * (
                2
                * (
                    2 * (2 * (2 * (2 * (2 * bitplane[0] + bitplane[1]) + bitplane[2]) + bitplane[3]) + bitplane[4])
                    + bitplane[5]
                )
                + bitplane[6]
            )
            + bitplane[7]
        )
        return result

    def calculate_complexity(self, img):
        counter = 0
        for r in range(img.shape[0]):
            for c in range(img.shape[1]):
                if r != img.shape[0] - 1:
                    if img[r][c] != img[r + 1][c]:
                        counter += 1
                if c != img.shape[1] - 1:
                    if img[r][c] != img[r][c + 1]:
                        counter += 1
        return counter / 112


def blocks_to_text(blocks):
    """Decode extracted 8x8 bit blocks until null terminator."""
    all_bits = ""
    for block in blocks:
        for bit in block.flatten():
            all_bits += str(int(bit))

    chars = []
    for i in range(0, len(all_bits), 8):
        byte = all_bits[i:i + 8]
        if len(byte) < 8:
            break
        val = int(byte, 2)
        if val == 0:
            break
        chars.append(chr(val))
    return "".join(chars)
