import numpy as np

class Message:
    @staticmethod
    def create_message(text):
        bits = ''.join([format(ord(i), '08b') for i in text])
        bits += '00000000'
        
        while len(bits) % 64 != 0:
            bits += '0'
            
        blocks = []
        for i in range(0, len(bits), 64):
            chunk = bits[i:i+64]
            bit_array = np.array([int(b) for b in chunk])
            blocks.append(bit_array.reshape(8, 8))
        return blocks