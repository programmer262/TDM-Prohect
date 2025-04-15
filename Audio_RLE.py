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

def rle_encode(data):
    """Run-Length Encoding for numerical data"""
    if not data:
        return []
    result = []
    count = 1
    current = data[0]
    for item in data[1:]:
        if item == current:
            count += 1
        else:
            result.append((current, count))
            current = item
            count = 1
    result.append((current, count))
    return result

def rle_decode(rle_data):
    """Decode Run-Length Encoded data"""
    result = []
    for value, count in rle_data:
        result.extend([value] * count)
    return result

def sauvegarder_compression(filename, qcoeffs, fs, frame_size):
    """
    Save compressed audio using Huffman coding followed by RLE
    
    Args:
        filename: Output file path
        qcoeffs: Quantized MDCT coefficients
        fs: Sampling frequency
        frame_size: MDCT frame size
    """
    # Initialize Huffman coder
    huffman = HuffmanCoder(precision=0)
    
    # Flatten and convert coefficients to list
    flat_qcoeffs = qcoeffs.flatten().tolist()
    
    # Apply Huffman coding
    encoded_bits, codes = huffman.encode(flat_qcoeffs)
    
    # Convert encoded bits to integer values for RLE
    encoded_ints = [int(bit) for bit in encoded_bits]
    
    # Apply RLE
    rle_data = rle_encode(encoded_ints)
    
    # Serialize RLE data
    rle_bytes = []
    for value, count in rle_data:
        rle_bytes.extend([value, count])
    rle_array = np.array(rle_bytes, dtype=np.int32)
    
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
        rle_array.tofile(f)
    
    print(f"Coefficients compressés avec Huffman + RLE et sauvegardés dans {filename}")

def charger_compression(filename):
    """
    Load and decompress audio file compressed with Huffman + RLE
    
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
        
        # Read RLE data
        rle_array = np.fromfile(f, dtype=np.int32)
    
    # Deserialize codes
    huffman = HuffmanCoder(precision=0)
    codes = huffman._deserialize_codes(codes_str)
    
    # Reconstruct RLE data
    rle_data = [(rle_array[i], rle_array[i+1]) for i in range(0, len(rle_array), 2)]
    
    # Decode RLE
    encoded_ints = rle_decode(rle_data)
    
    # Convert back to bits
    encoded_bits = ''.join(str(bit) for bit in encoded_ints)
    
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

    # Sauvegarde compressée avec Huffman + RLE
    after = "output.irm"
    sauvegarder_compression(after, qcoeffs, fs, frame_size)

    # Décompression
    qcoeffs_loaded, fs_loaded, frame_size_loaded = charger_compression(after)
    coeffs_recon = dequantifier(qcoeffs_loaded, q=q)
    signal_recon = imdct(coeffs_recon, frame_size=frame_size_loaded)
    signal_recon = normaliser_signal(signal_recon)
    
    # Audio reconstruit (16 bits)
    audio_recon = signal_to_audio(signal_recon, fs)
    audio_recon.export("reconstructed.wav", format="wav")

    # Audio 8 bits
    signal_8bit = requantifier_signal(signal_recon, bits=8)
    print("Signal 8-bit - Max:", max(signal_8bit), "Min:", min(signal_8bit))
    
    # Compression Huffman + RLE du signal 8-bit
    huffman_8bit = HuffmanCoder(precision=0)
    encoded_8bit, codes_8bit = huffman_8bit.encode(signal_8bit.tolist())
    encoded_ints_8bit = [int(bit) for bit in encoded_8bit]
    rle_8bit = rle_encode(encoded_ints_8bit)
    rle_bytes_8bit = np.array([val for pair in rle_8bit for val in pair], dtype=np.int32)
    codes_8bit_str = huffman_8bit._serialize_codes(codes_8bit)
    codes_8bit_bytes = codes_8bit_str.encode('utf-8')
    
    # Sauvegarde du signal 8-bit compressé
    with open("reconstructed_8bit.irm", 'wb') as f:
        header = np.array([fs, len(codes_8bit_bytes)], dtype=np.int32)
        header.tofile(f)
        f.write(codes_8bit_bytes)
        rle_bytes_8bit.tofile(f)
    
    # Version non-compressée pour comparaison
    pcm_8bit = signal_8bit.tobytes()
    audio_8bit = AudioSegment(pcm_8bit, frame_rate=fs, sample_width=1, channels=1)
    audio_8bit.export("reconstructed_8bit.wav", format="wav")

    # Affichage des tailles
    afficher_taille_fichiers(before, "reconstructed_8bit.irm")

    # Lecture et tracé
    play(audio_8bit)
    plt.figure(figsize=(12, 6))
    plt.plot(signal[:400], label="Original")
    plt.plot(signal_recon[:400], label="Reconstruit", alpha=0.7)
    plt.legend()
    plt.title("Comparaison Original vs Reconstruit")
    plt.show()

if __name__ == "__main__":
    preparer_et_comprimer()