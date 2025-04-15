import numpy as np

class LZ78Compressor:
    def __init__(self, precision=2):
        """
        Initialize LZ78 compressor
        :param precision: number of decimal places to round to (0 for integers)
        """
        self.precision = precision

    def _process_input(self, data):
        """Convert input to consistent format, handling numpy arrays"""
        if isinstance(data, np.ndarray):
            data = data.flatten()
            if self.precision == 0:
                return [int(round(x)) for x in data]
            else:
                return [round(float(x), self.precision) for x in data]
        elif isinstance(data, (list, tuple)):
            if self.precision == 0:
                return [int(round(x)) if isinstance(x, (float, np.floating)) else x for x in data]
            else:
                return [round(float(x), self.precision) if isinstance(x, (float, np.floating)) else x for x in data]
        else:
            raise TypeError("Input must be list, tuple, or numpy array")

    def compress(self, data):
        """LZ78 compression that handles numpy arrays"""
        processed_data = self._process_input(data)
        
        dictionary = {0: []}  # Initialize with empty list
        result = []
        current_sequence = []
        next_code = 1
        
        for symbol in processed_data:
            # Create new candidate sequence
            candidate_sequence = current_sequence.copy()
            candidate_sequence.append(symbol)
            
            # Check if this sequence exists in dictionary
            sequence_exists = False
            for code, seq in dictionary.items():
                if seq == candidate_sequence:
                    current_sequence = candidate_sequence
                    sequence_exists = True
                    break
            
            if not sequence_exists:
                # Find the code for the longest existing prefix
                prefix_code = 0
                for i in range(1, len(current_sequence)+1):
                    prefix = current_sequence[:i]
                    for code, seq in dictionary.items():
                        if seq == prefix:
                            prefix_code = code
                            break
                
                # Add new sequence to dictionary
                new_symbol = symbol if not current_sequence else current_sequence[-1]
                result.append((prefix_code, new_symbol))
                dictionary[next_code] = candidate_sequence
                next_code += 1
                current_sequence = []
        
        # Handle any remaining sequence
        if current_sequence:
            prefix_code = 0
            for i in range(1, len(current_sequence)+1):
                prefix = current_sequence[:i]
                for code, seq in dictionary.items():
                    if seq == prefix:
                        prefix_code = code
                        break
            result.append((prefix_code, ""))
        
        return result

    def decompress(self, compressed_data):
        """LZ78 decompression that properly handles numerical data"""
        dictionary = {0: []}
        result = []
        next_code = 1
        
        for code, symbol in compressed_data:
            # Get the prefix sequence
            sequence = dictionary[code].copy()
            
            # Add the new symbol if present
            if symbol != "":
                sequence.append(symbol)
            
            # Add to result and dictionary
            result.extend(sequence)
            dictionary[next_code] = sequence
            next_code += 1
        
        # Convert to numpy array with proper type handling
        if all(isinstance(x, (int, float)) for x in result):
            arr = np.array(result)
            if self.precision == 0:
                return arr.astype(np.int64)
            else:
                return np.round(arr, self.precision)
        return np.array(result)

    def test_compression(self, data):

        
        compressed = self.compress(data)
        
        decompressed = self.decompress(compressed)
        
        # Prepare comparison
        if isinstance(data, np.ndarray):
            original_processed = np.round(data, self.precision) if self.precision > 0 else data.astype(np.int64)
            success = np.array_equal(original_processed, decompressed)
        else:
            original_processed = [round(x, self.precision) if isinstance(x, float) else x for x in data]
            success = original_processed == decompressed.tolist()
        
        print("Test", "PASSED" if success else "E1")
        return success


