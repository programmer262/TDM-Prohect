import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from pydub import AudioSegment
from pydub.playback import play
from scipy.signal import butter, filtfilt
from scipy.fftpack import dct, idct
import os
import matplotlib.pyplot as plt
from huffman import HuffmanCoder  # Import HuffmanCoder class

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
    Save compressed audio using Huffman coding
    
    Args:
        filename: Output file path
        qcoeffs: Quantized MDCT coefficients
        fs: Sampling frequency
        frame_size: MDCT frame size
    """
    # Initialize Huffman coder
    huffman = HuffmanCoder(precision=0)  # precision=0 for int16 data
    
    # Flatten the coefficients array for compression
    flat_qcoeffs = qcoeffs.flatten().tolist()
    
    # Compress using Huffman coding
    compressed_data = huffman.compress(flat_qcoeffs)
    
    # Save header information separately (not compressed)
    with open(filename, 'wb') as f:
        # Write header: fs, frame_size, original shape
        header = np.array([fs, frame_size, qcoeffs.shape[0], qcoeffs.shape[1]], dtype=np.int32)
        header.tofile(f)
        
        # Write compressed data
        f.write(compressed_data)
    
    print(f"Coefficients compressés avec Huffman et sauvegardés dans {filename}")

def charger_compression(filename):
    """
    Load and decompress audio file compressed with Huffman coding
    
    Args:
        filename: Input file path
        
    Returns:
        qcoeffs: Decompressed quantized coefficients
        fs: Sampling frequency
        frame_size: MDCT frame size
    """
    with open(filename, 'rb') as f:
        # Read header
        header = np.fromfile(f, dtype=np.int32, count=4)
        fs, frame_size, rows, cols = header
        
        # Read remaining bytes (compressed data)
        compressed_data = f.read()
    
    # Initialize Huffman decoder
    huffman = HuffmanCoder(precision=0)
    
    # Decompress data
    flat_qcoeffs = huffman.decompress(compressed_data)
    
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
    #signal = filtrer_frequences_inaudibles(signal, fs)
    signal = normaliser_signal(signal)

    # MDCT
    frame_size = 1024
    coeffs = mdct(signal, frame_size=frame_size)
    print("MDCT calculée :", coeffs.shape)

    # Quantification
    q = 0.02
    qcoeffs = quantifier(coeffs, q=q)

    # Sauvegarde compressée avec Huffman
    after = "output.irm"
    sauvegarder_compression(after, qcoeffs, fs, frame_size)

    # Décompression avec Huffman
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
    
    # Compression Huffman du signal 8-bit
    huffman_8bit = HuffmanCoder(precision=0)
    compressed_8bit = huffman_8bit.compress(signal_8bit.tolist())
    
    # Sauvegarde du signal 8-bit compressé
    with open("reconstructed_8bit.irm", 'wb') as f:
        np.array([fs], dtype=np.int32).tofile(f)
        f.write(compressed_8bit)
    
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