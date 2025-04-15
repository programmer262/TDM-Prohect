import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from pydub import AudioSegment
from pydub.playback import play
from scipy.signal import butter, filtfilt
from scipy.fftpack import dct, idct
import os
import matplotlib.pyplot as plt
import struct

class LZWCoder:
    """
    Implementation of the LZW (Lempel-Ziv-Welch) compression algorithm
    """
    
    def __init__(self, max_dict_size=32768):  # 2^15 dictionary size by default
        """
        Initialize the LZW coder with a maximum dictionary size
        """
        self.max_dict_size = min(max_dict_size, 65535)  # Ensure it fits in 16 bits
        
    def compress(self, data):
        """
        Compress bytes using LZW algorithm
        :param data: bytes object to compress
        :return: compressed bytes
        """
        if not data:
            return b''
        
        # Add header with max dictionary size
        result = bytearray(struct.pack('>H', self.max_dict_size))
        
        # Initialize dictionary with single bytes (0-255)
        dictionary = {bytes([i]): i for i in range(256)}
        next_code = 256
        
        buffer = bytearray()  # Current sequence being processed
        
        # Use variable-length code encoding
        for byte in data:
            current = buffer + bytes([byte])
            
            if bytes(current) in dictionary:
                buffer = current
            else:
                # Output code for buffer
                code = dictionary[bytes(buffer)]
                
                # Determine bytes needed to represent this code
                if code < 256:
                    result.append(0)  # 0 prefix means 1-byte code
                    result.append(code)
                elif code < 65536:
                    result.append(1)  # 1 prefix means 2-byte code
                    result.extend(struct.pack('>H', code))
                else:
                    result.append(2)  # 2 prefix means 4-byte code
                    result.extend(struct.pack('>I', code))
                
                # Add new sequence to dictionary if space allows
                if next_code < self.max_dict_size:
                    dictionary[bytes(current)] = next_code
                    next_code += 1
                
                buffer = bytearray([byte])
        
        # Output code for any remaining buffer
        if buffer:
            code = dictionary[bytes(buffer)]
            if code < 256:
                result.append(0)
                result.append(code)
            elif code < 65536:
                result.append(1)
                result.extend(struct.pack('>H', code))
            else:
                result.append(2)
                result.extend(struct.pack('>I', code))
        
        return bytes(result)
    
    def decompress(self, compressed_data):
        """
        Decompress LZW-compressed data
        :param compressed_data: bytes object containing LZW-compressed data
        :return: decompressed bytes
        """
        if len(compressed_data) < 2:
            return b''
        
        # Read header with dictionary size
        max_dict_size = struct.unpack('>H', compressed_data[:2])[0]
        self.max_dict_size = max_dict_size
        
        pos = 2  # Start after header
        
        # Initialize dictionary for decompression
        dictionary = {i: bytes([i]) for i in range(256)}
        next_code = 256
        
        result = bytearray()
        old_code = None
        
        while pos < len(compressed_data):
            # Read code type prefix
            code_type = compressed_data[pos]
            pos += 1
            
            # Read code based on type
            if code_type == 0:  # 1-byte code
                if pos >= len(compressed_data):
                    break
                code = compressed_data[pos]
                pos += 1
            elif code_type == 1:  # 2-byte code
                if pos + 1 >= len(compressed_data):
                    break
                code = struct.unpack('>H', compressed_data[pos:pos+2])[0]
                pos += 2
            elif code_type == 2:  # 4-byte code
                if pos + 3 >= len(compressed_data):
                    break
                code = struct.unpack('>I', compressed_data[pos:pos+4])[0]
                pos += 4
            else:
                # Invalid code type
                break
            
            # First code is always a known symbol
            if old_code is None:
                result.extend(dictionary[code])
                old_code = code
                continue
            
            # Get entry from dictionary or handle special case
            if code in dictionary:
                entry = dictionary[code]
                result.extend(entry)
                # Add new entry to dictionary
                if next_code < max_dict_size:
                    dictionary[next_code] = dictionary[old_code] + bytes([entry[0]])
                    next_code += 1
            else:
                # Special case: code not in dictionary yet
                entry = dictionary[old_code] + bytes([dictionary[old_code][0]])
                result.extend(entry)
                # Add new entry to dictionary
                if next_code < max_dict_size:
                    dictionary[next_code] = entry
                    next_code += 1
            
            old_code = code
        
        return bytes(result)