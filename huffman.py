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

class HuffmanNode:
    def __init__(self, symbol=None, frequency=0, left=None, right=None):
        self.symbol = symbol
        self.frequency = frequency
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.frequency < other.frequency

    def is_leaf(self):
        return self.left is None and self.right is None


class HuffmanCoder:

    def __init__(self, precision=0):
        """
        Initialize Huffman coder
        :param precision: number of decimal places to round to (0 for integers)
        """
        self.precision = precision
        self.codes = {}
        self.tree = None

    def _calculate_frequencies(self, data):
        """Calculate symbol frequencies in O(n) time, handles numpy arrays"""
        freq = {}
        
        # Convert numpy array to list if needed
        if isinstance(data, np.ndarray):
            data = data.flatten().tolist()
        
        total = len(data)
        if total == 0:
            return freq

        # Handle numerical data rounding
        if isinstance(data[0], (int, float, np.integer, np.floating)):
            if self.precision == 0:
                data = [int(round(x)) for x in data]
            else:
                data = [round(float(x), self.precision) for x in data]

        for symbol in data:
            if symbol in freq:
                freq[symbol] += 1
            else:
                freq[symbol] = 1

        # Normalize frequencies
        return {k: v/total for k, v in freq.items()}

    def encode(self, data):
        """Encode input data using Huffman coding, handles numpy arrays"""
        if isinstance(data, np.ndarray):
            data = data.flatten().tolist()
        
        if not data:
            return "", {}

        frequencies = self._calculate_frequencies(data)
        self.tree = self.build_tree(frequencies)
        self.codes = {}
        
        if self.tree:
            self._generate_codes(self.tree)
        
        # Special case: all symbols identical
        if not self.codes and data:
            self.codes = {data[0]: "0"}
        
        encoded_bits = []
        for symbol in data:
            # Handle numerical data rounding
            if isinstance(symbol, (int, float, np.integer, np.floating)):
                if self.precision == 0:
                    symbol = int(round(symbol))
                else:
                    symbol = round(float(symbol), self.precision)
            encoded_bits.append(self.codes[symbol])
        
        return "".join(encoded_bits), self.codes


    def _build_min_heap(self, frequencies):
        """Build a min-heap from frequencies"""
        heap = []
        for symbol, freq in frequencies.items():
            heap.append(HuffmanNode(symbol=symbol, frequency=freq))
        
        # Heapify - simple bubble down approach
        for i in range(len(heap)//2 - 1, -1, -1):
            self._heapify(heap, i)
        return heap

    def _heapify(self, heap, index):
        """Maintain min-heap property starting at index"""
        smallest = index
        left = 2 * index + 1
        right = 2 * index + 2

        if left < len(heap) and heap[left] < heap[smallest]:
            smallest = left
        if right < len(heap) and heap[right] < heap[smallest]:
            smallest = right

        if smallest != index:
            heap[index], heap[smallest] = heap[smallest], heap[index]
            self._heapify(heap, smallest)

    def _extract_min(self, heap):
        """Extract and return the minimum node, maintaining heap"""
        if not heap:
            return None
        
        # Swap root with last element
        heap[0], heap[-1] = heap[-1], heap[0]
        min_node = heap.pop()
        
        # Heapify down
        if heap:
            self._heapify(heap, 0)
        return min_node

    def _insert_heap(self, heap, node):
        """Insert a node into the min-heap"""
        heap.append(node)
        # Bubble up
        index = len(heap) - 1
        parent = (index - 1) // 2
        
        while index > 0 and heap[index] < heap[parent]:
            heap[index], heap[parent] = heap[parent], heap[index]
            index = parent
            parent = (index - 1) // 2

    def build_tree(self, frequencies):
        """Build Huffman tree from frequencies"""
        if not frequencies:
            return None

        heap = self._build_min_heap(frequencies)
        
        while len(heap) > 1:
            left = self._extract_min(heap)
            right = self._extract_min(heap)
            
            merged = HuffmanNode(
                frequency=left.frequency + right.frequency,
                left=left,
                right=right
            )
            self._insert_heap(heap, merged)
        
        return heap[0] if heap else None

    def _generate_codes(self, node, current_code=""):
        """Recursively generate Huffman codes"""
        if node.is_leaf():
            self.codes[node.symbol] = current_code or "0"  # Ensure at least one bit
        else:
            if node.left:
                self._generate_codes(node.left, current_code + "0")
            if node.right:
                self._generate_codes(node.right, current_code + "1")

    def decode(self, encoded_bits, code_table=None):
        """Decode Huffman encoded bits back to original data"""
        if not encoded_bits:
            return []
        
        if code_table is None:
            code_table = self.codes
        
        # Build decoding tree if not available
        if not code_table:
            raise ValueError("No code table provided for decoding")
        
        # Reverse code table for decoding
        code_to_symbol = {v: k for k, v in code_table.items()}
        
        decoded_data = []
        current_code = ""
        
        for bit in encoded_bits:
            current_code += bit
            if current_code in code_to_symbol:
                decoded_data.append(code_to_symbol[current_code])
                current_code = ""
        
        return decoded_data

    def _serialize_codes(self, codes):
        """Safely serialize Huffman codes dictionary to string"""
        items = []
        for symbol, code in codes.items():
            # Handle different symbol types
            if isinstance(symbol, int):
                type_char = 'i'
            elif isinstance(symbol, float):
                type_char = 'f'
            else:  # string
                type_char = 's'
            items.append(f"{type_char}{symbol}:{code}")
        return '|'.join(items)

    def _deserialize_codes(self, codes_str):
        """Deserialize string back to Huffman codes dictionary"""
        codes = {}
        if not codes_str:
            return codes
            
        for item in codes_str.split('|'):
            if not item or ':' not in item:
                continue
                
            type_char = item[0]
            symbol_part, code = item[1:].split(':', 1)
            
            try:
                if type_char == 'i':
                    symbol = int(symbol_part)
                elif type_char == 'f':
                    symbol = float(symbol_part)
                else:  # 's'
                    symbol = symbol_part
                codes[symbol] = code
            except ValueError:
                continue
        return codes

    def package(self, data):
        """Package encoded data and codes into a single binary format"""
        # 1. Encode the data
        encoded_bits, codes = self.encode(data)
        
        # 2. Serialize codes dictionary safely
        codes_str = self._serialize_codes(codes)
        codes_bytes = codes_str.encode('utf-8')
        
        # 3. Pad the encoded bits to complete bytes
        padding = (8 - len(encoded_bits) % 8) % 8
        padded_encoded = encoded_bits + '0' * padding
        
        # 4. Convert encoded bits to bytes
        encoded_bytes = bytes(int(padded_encoded[i:i+8], 2) 
                            for i in range(0, len(padded_encoded), 8))
        
        # 5. Create fixed-size header (10 bytes)
        header = bytes([
            len(codes_bytes) >> 24 & 0xFF,
            len(codes_bytes) >> 16 & 0xFF,
            len(codes_bytes) >> 8 & 0xFF,
            len(codes_bytes) & 0xFF,
            len(encoded_bits) >> 24 & 0xFF,
            len(encoded_bits) >> 16 & 0xFF,
            len(encoded_bits) >> 8 & 0xFF,
            len(encoded_bits) & 0xFF,
            padding,
            self.precision
        ])
        
        # 6. Combine everything
        return header + codes_bytes + encoded_bytes

    def unpackage(self, package):
        """Extract encoded data and codes from packaged binary"""
        if len(package) < 10:
            raise ValueError("Invalid package format")
        
        # 1. Parse fixed-size header (10 bytes)
        codes_len = (package[0] << 24) | (package[1] << 16) | (package[2] << 8) | package[3]
        encoded_len = (package[4] << 24) | (package[5] << 16) | (package[6] << 8) | package[7]
        padding = package[8]
        precision = package[9]
        
        # 2. Extract codes
        codes_start = 10
        codes_end = codes_start + codes_len
        if codes_end > len(package):
            raise ValueError("Corrupted package - codes length mismatch")
            
        codes_str = package[codes_start:codes_end].decode('utf-8')
        codes = self._deserialize_codes(codes_str)
        
        # 3. Extract encoded data
        encoded_bytes = package[codes_end:]
        
        # 4. Convert back to binary string
        encoded_bits = ''.join(f'{byte:08b}' for byte in encoded_bytes)
        encoded_bits = encoded_bits[:encoded_len]  # Remove padding
        
        return encoded_bits, codes, precision

    def compress(self, data):
        """Full compression pipeline"""
        return self.package(data)

    def decompress(self, package):
        """Full decompression pipeline"""
        encoded_bits, codes, precision = self.unpackage(package)
        self.precision = precision  # Restore precision
        return self.decode(encoded_bits, codes)