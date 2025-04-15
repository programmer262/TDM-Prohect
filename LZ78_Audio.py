import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from pydub import AudioSegment
from pydub.playback import play
from scipy.signal import butter, filtfilt
from scipy.fftpack import dct, idct
import os
import matplotlib.pyplot as plt
from huffman import HuffmanCoder

class LZ78Compressor:
    def __init__(self, precision=0):
        self.precision = precision

    def _process_input(self, data):
        if isinstance(data, np.ndarray):
            data = data.flatten()
            if self.precision == 0:
                return [int(round(x)) for x in data]
            return [round(float(x), self.precision) for x in data]
        elif isinstance(data, (list, tuple)):
            if self.precision == 0:
                return [int(round(x)) if isinstance(x, (float, np.floating)) else x for x in data]
            return [round(float(x), self.precision) if isinstance(x, (float, np.floating)) else x for x in data]
        raise TypeError("Input must be list, tuple, or numpy array")

    def compress(self, data):
        processed_data = self._process_input(data)
        dictionary = {0: []}
        result = []
        current_sequence = []
        next_code = 1
        
        for symbol in processed_data:
            candidate_sequence = current_sequence.copy()
            candidate_sequence.append(symbol)
            sequence_exists = False
            for code, seq in dictionary.items():
                if seq == candidate_sequence:
                    current_sequence = candidate_sequence
                    sequence_exists = True
                    break
            if not sequence_exists:
                prefix_code = 0
                for i in range(1, len(current_sequence)+1):
                    prefix = current_sequence[:i]
                    for code, seq in dictionary.items():
                        if seq == prefix:
                            prefix_code = code
                            break
                result.append((prefix_code, symbol))
                dictionary[next_code] = candidate_sequence
                next_code += 1
                current_sequence = []
        
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
        dictionary = {0: []}
        result = []
        next_code = 1
        
        for code, symbol in compressed_data:
            sequence = dictionary[code].copy()
            if symbol != "":
                sequence.append(symbol)
            result.extend(sequence)
            dictionary[next_code] = sequence
            next_code += 1
        
        if all(isinstance(x, (int, float)) for x in result):
            arr = np.array(result)
            if self.precision == 0:
                return arr.astype(np.int64)
            return np.round(arr, self.precision)
        return np.array(result)

def file():
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    file_path = askopenfilename(title="Sélectionnez un fichier audio")
    root.destroy()
    return file_path

def charger_fichier():
    file_path = file()
    audio = AudioSegment.from_file(file_path).set_channels(1)
    return audio, file_path

def centrer_signal(signal):
    return signal - np.mean(signal)

def normaliser_signal(signal):
    max_abs = np.max(np.abs(signal))
    return signal / max_abs if max_abs != 0 else signal

def filtrer_frequences_inaudibles(signal, fs, f_min=20, f_max=20000):
    nyq = 0.5 * fs
    f_max = min(f_max, nyq)
    f_min = max(1, min(f_min, nyq - 1))
    b, a = butter(N=4, Wn=[f_min / nyq, f_max / nyq], btype='band')
    return filtfilt(b, a, signal)

def audio_to_signal(audio):
    signal = np.array(audio.get_array_of_samples()).astype(np.float32)
    return signal / max(abs(signal))

def mdct(signal, frame_size=1024):
    N = frame_size
    overlap = N // 2
    nb_frames = (len(signal) - overlap) // overlap
    mdct_coeffs = []
    window = np.sin(np.pi / N * (np.arange(N) + 0.5))
    for i in range(nb_frames):
        frame = signal[i * overlap : i * overlap + N]
        if len(frame) < N:
            break
        windowed = frame * window
        coeffs = dct(windowed, type=2, norm='ortho')
        mdct_coeffs.append(coeffs)
    return np.array(mdct_coeffs)

def imdct(coeffs, frame_size=1024):
    N = frame_size
    overlap = N // 2
    window = np.sin(np.pi / N * (np.arange(N) + 0.5))
    signal = np.zeros(overlap * (len(coeffs) + 1))
    for i, c in enumerate(coeffs):
        frame = idct(c, type=2, norm='ortho') * window
        start = i * overlap
        signal[start:start+N] += frame
    return signal

def quantifier(coeffs, q=0.02):
    return np.round(coeffs / q).astype(np.int16)

def dequantifier(qcoeffs, q=0.02):
    return qcoeffs.astype(np.float32) * q

def sauvegarder_compression(filename, qcoeffs, fs, frame_size):
    """
    Save compressed audio using Huffman coding followed by LZ78
    
    Args:
        filename: Output file path
        qcoeffs: Quantized MDCT coefficients
        fs: Sampling frequency
        frame_size: MDCT frame size
    """
    # Initialize Huffman coder
    huffman = HuffmanCoder(precision=0)
    lz78 = LZ78Compressor(precision=0)
    
    # Flatten and convert coefficients to list
    flat_qcoeffs = qcoeffs.flatten().tolist()
    
    # Apply Huffman coding
    encoded_bits, codes = huffman.encode(flat_qcoeffs)
    
    # Convert encoded bits to integers for LZ78
    encoded_ints = [int(bit) for bit in encoded_bits]
    
    # Apply LZ78 compression
    lz78_data = lz78.compress(encoded_ints)
    
    # Serialize LZ78 data
    lz78_flat = []
    for prefix, symbol in lz78_data:
        lz78_flat.append(prefix)
        lz78_flat.append(int(symbol) if symbol != "" else -1)  # Use -1 for empty symbol
    lz78_array = np.array(lz78_flat, dtype=np.int32)
    
    # Convert codes to bytes
    codes_str = huffman._serialize_codes(codes)
    codes_bytes = codes_str.encode('utf-8')
    
    # Create header
    header = np.array([
        fs, frame_size, 
        qcoeffs.shape[0], qcoeffs.shape[1], 
        len(codes_bytes)
    ], dtype=np.int32)
    
    # Save to file
    with open(filename, 'wb') as f:
        header.tofile(f)
        f.write(codes_bytes)
        lz78_array.tofile(f)
    
    print(f"Coefficients compressés avec Huffman + LZ78 et sauvegardés dans {filename}")

def charger_compression(filename):
    """
    Load and decompress audio file compressed with Huffman + LZ78
    
    Args:
        filename: Input file path
        
    Returns:
        qcoeffs: Decompressed quantized coefficients
        fs: Sampling frequency
        frame_size: MDCT frame size
    """
    with open(filename, 'rb') as f:
        # Read header
        header = np.fromfile(f, dtype=np.int32, count=5)
        fs, frame_size, rows, cols, codes_len = header
        
        # Read codes
        codes_bytes = f.read(codes_len)
        codes_str = codes_bytes.decode('utf-8')
        
        # Read LZ78 data
        lz78_array = np.fromfile(f, dtype=np.int32)
    
    # Deserialize codes
    huffman = HuffmanCoder(precision=0)
    codes = huffman._deserialize_codes(codes_str)
    
    # Reconstruct LZ78 data
    lz78_data = [(lz78_array[i], str(lz78_array[i+1]) if lz78_array[i+1] != -1 else "") 
                 for i in range(0, len(lz78_array), 2)]
    
    # Decode LZ78
    lz78 = LZ78Compressor(precision=0)
    encoded_ints = lz78.decompress(lz78_data)
    
    # Convert back to bits
    encoded_bits = ''.join(str(int(x)) for x in encoded_ints)
    
    # Decode Huffman
    flat_qcoeffs = huffman.decode(encoded_bits, codes)
    
    # Reshape to original dimensions
    qcoeffs = np.array(flat_qcoeffs, dtype=np.int16).reshape(rows, cols)
    
    print(f"Fichier décompressé: {rows} frames de taille {cols}")
    return qcoeffs, fs, frame_size

def signal_to_audio(signal, fs):
    signal = np.clip(signal, -1, 1)
    pcm = (signal * 32767).astype(np.int16)
    audio = AudioSegment(pcm.tobytes(), frame_rate=fs, sample_width=2, channels=1)
    return audio

def requantifier_signal(signal, bits=8):
    L = 2 ** bits
    signal_scaled = (signal + 1) / 2
    signal_quantized = np.round(signal_scaled * (L - 1))
    return signal_quantized.astype(np.uint8)

def afficher_taille_fichiers(before, after):
    ta = os.path.getsize(before) / 1024
    tb = os.path.getsize(after) / 1024
    print(f"Taille avant : {ta:.2f} Ko")
    print(f"Taille compressée : {tb:.2f} Ko")
    print(f"Taux de compression : {ta/tb:.2f}x")

def preparer_et_comprimer():
    audio, path = charger_fichier()
    print("Traitement du fichier :", path)
    before = "original.wav"
    audio.export(before, format="wav")

    fs = audio.frame_rate
    signal = audio_to_signal(audio)
    signal = centrer_signal(signal)
    signal = normaliser_signal(signal)

    # MDCT
    frame_size = 1024
    coeffs = mdct(signal, frame_size=frame_size)
    print("MDCT calculée :", coeffs.shape)

    # Quantification
    q = 0.02
    qcoeffs = quantifier(coeffs, q=q)

    # Sauvegarde compressée avec Huffman + LZ78
    after = "output_lz78.irm"
    sauvegarder_compression(after, qcoeffs, fs, frame_size)

    # Décompression
    qcoeffs_loaded, fs_loaded, frame_size_loaded = charger_compression(after)
    coeffs_recon = dequantifier(qcoeffs_loaded, q=q)
    signal_recon = imdct(coeffs_recon, frame_size=frame_size_loaded)
    signal_recon = normaliser_signal(signal_recon)
    
    # Audio reconstruit (16 bits)
    audio_recon = signal_to_audio(signal_recon, fs)
    audio_recon.export("reconstructed_lz78.wav", format="wav")

    # Audio 8 bits
    signal_8bit = requantifier_signal(signal_recon, bits=8)
    print("Signal 8-bit - Max:", max(signal_8bit), "Min:", min(signal_8bit))
    
    # Compression Huffman + LZ78 du signal 8-bit
    huffman_8bit = HuffmanCoder(precision=0)
    lz78_8bit = LZ78Compressor(precision=0)
    encoded_8bit, codes_8bit = huffman_8bit.encode(signal_8bit.tolist())
    encoded_ints_8bit = [int(bit) for bit in encoded_8bit]
    lz78_8bit_data = lz78_8bit.compress(encoded_ints_8bit)
    lz78_8bit_flat = []
    for prefix, symbol in lz78_8bit_data:
        lz78_8bit_flat.append(prefix)
        lz78_8bit_flat.append(int(symbol) if symbol != "" else -1)
    lz78_8bit_array = np.array(lz78_8bit_flat, dtype=np.int32)
    codes_8bit_str = huffman_8bit._serialize_codes(codes_8bit)
    codes_8bit_bytes = codes_8bit_str.encode('utf-8')
    
    # Sauvegarde du signal 8-bit compressé
    with open("reconstructed_8bit_lz78.irm", 'wb') as f:
        header = np.array([fs, len(codes_8bit_bytes)], dtype=np.int32)
        header.tofile(f)
        f.write(codes_8bit_bytes)
        lz78_8bit_array.tofile(f)
    
    # Version non-compressée pour comparaison
    pcm_8bit = signal_8bit.tobytes()
    audio_8bit = AudioSegment(pcm_8bit, frame_rate=fs, sample_width=1, channels=1)
    audio_8bit.export("reconstructed_8bit_lz78.wav", format="wav")

    # Affichage des tailles
    afficher_taille_fichiers(before, "reconstructed_8bit_lz78.irm")

    # Lecture et tracé
    play(audio_8bit)
    plt.figure(figsize=(12, 6))
    plt.plot(signal[:400], label="Original")
    plt.plot(signal_recon[:400], label="Reconstruit", alpha=0.7)
    plt.legend()
    plt.title("Comparaison Original vs Reconstruit (Huffman + LZ78)")
    plt.show()

if __name__ == "__main__":
    preparer_et_comprimer()