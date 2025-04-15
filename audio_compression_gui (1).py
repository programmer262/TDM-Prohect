import sys
import os
import numpy as np
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                              QPushButton, QFileDialog, QLabel, QComboBox, QSizePolicy)
from pydub import AudioSegment
from pydub.playback import play
from scipy.signal import butter, filtfilt
from scipy.fftpack import dct, idct
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from huffman import HuffmanCoder
from LZW import LZWCoder
from psychoacousticmodel import PsychoacousticModel
import warnings
warnings.filterwarnings('ignore')

# Global variables
audio = None
file_path = ""
compressed_file = "output.irm"
original_file = "original.wav"
reconstructed_file = "reconstructed.wav"

def centrer_signal(signal):
    return signal - np.mean(signal)

def normaliser_signal(signal):
    max_abs = np.max(np.abs(signal))
    return signal / max_abs if max_abs != 0 else signal

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

def compress_audio():
    global audio, file_path, layout
    if not audio:
        file_label.setText("Aucun fichier audio chargé")
        return
    
    compression_type = compression_combo.currentText()
    
    # Load and preprocess audio
    audio.export(original_file, format="wav")
    fs = audio.frame_rate
    signal = audio_to_signal(audio)
    signal = centrer_signal(signal)
    signal = normaliser_signal(signal)
    
    frame_size = 1024
    coeffs = mdct(signal, frame_size=frame_size)
    
    # Initialize PsychoacousticModel if masking is involved

    q = 0.02
    qcoeffs = quantifier(coeffs, q=q)
    
    # Compression based on selected type
    if compression_type == "LZW only":
        lzw = LZWCoder()
        compressed_data = lzw.compress(qcoeffs.flatten().tobytes())
    elif compression_type == "Huffman only":
        huffman = HuffmanCoder(precision=0)
        flat_qcoeffs = qcoeffs.flatten().tolist()
        compressed_data = huffman.compress(flat_qcoeffs)
    elif compression_type == "Huffman + LZW":
        huffman = HuffmanCoder(precision=0)
        flat_qcoeffs = qcoeffs.flatten().tolist()
        huffman_compressed = huffman.compress(flat_qcoeffs)
        lzw = LZWCoder()
        compressed_data = lzw.compress(huffman_compressed)
    elif compression_type == "Huffman + LZW + Masquage":
        psycho_model = PsychoacousticModel(fs)
        weights = psycho_model.perceptual_bit_allocation(coeffs, frame_size)
        coeffs = coeffs * weights
    
        huffman = HuffmanCoder(precision=0)
        flat_qcoeffs = qcoeffs.flatten().tolist()
        huffman_compressed = huffman.compress(flat_qcoeffs)
        lzw = LZWCoder()
        compressed_data = lzw.compress(huffman_compressed)
    elif compression_type == "Huffman + Masquage":
        psycho_model = PsychoacousticModel(fs)
        weights = psycho_model.perceptual_bit_allocation(coeffs, frame_size)
        coeffs = coeffs * weights
    
        huffman = HuffmanCoder(precision=0)
        flat_qcoeffs = qcoeffs.flatten().tolist()
        compressed_data = huffman.compress(flat_qcoeffs)
    elif compression_type == "LZW + Masquage":
        psycho_model = PsychoacousticModel(fs)
        weights = psycho_model.perceptual_bit_allocation(coeffs, frame_size)
        coeffs = coeffs * weights
    
        lzw = LZWCoder()
        compressed_data = lzw.compress(qcoeffs.flatten().tobytes())
    
    # Save compressed data
    with open(compressed_file, 'wb') as f:
        header = np.array([fs, frame_size, qcoeffs.shape[0], qcoeffs.shape[1]], dtype=np.int32)
        header.tofile(f)
        f.write(compressed_data)
    
    # Decompression
    with open(compressed_file, 'rb') as f:
        header = np.fromfile(f, dtype=np.int32, count=4)
        fs_loaded, frame_size_loaded, rows, cols = header
        compressed_data = f.read()
    
    if compression_type == "LZW only":
        lzw = LZWCoder()
        decompressed_data = lzw.decompress(compressed_data)
        qcoeffs_loaded = np.frombuffer(decompressed_data, dtype=np.int16).reshape(rows, cols)
    elif compression_type == "Huffman only":
        huffman = HuffmanCoder(precision=0)
        flat_qcoeffs = huffman.decompress(compressed_data)
        qcoeffs_loaded = np.array(flat_qcoeffs, dtype=np.int16).reshape(rows, cols)
    elif compression_type == "Huffman + LZW":
        lzw = LZWCoder()
        huffman_data = lzw.decompress(compressed_data)
        huffman = HuffmanCoder(precision=0)
        flat_qcoeffs = huffman.decompress(huffman_data)
        qcoeffs_loaded = np.array(flat_qcoeffs, dtype=np.int16).reshape(rows, cols)
    elif compression_type == "Huffman + LZW + Masquage":
        lzw = LZWCoder()
        huffman_data = lzw.decompress(compressed_data)
        huffman = HuffmanCoder(precision=0)
        flat_qcoeffs = huffman.decompress(huffman_data)
        qcoeffs_loaded = np.array(flat_qcoeffs, dtype=np.int16).reshape(rows, cols)
    elif compression_type == "Huffman + Masquage":
        huffman = HuffmanCoder(precision=0)
        flat_qcoeffs = huffman.decompress(compressed_data)
        qcoeffs_loaded = np.array(flat_qcoeffs, dtype=np.int16).reshape(rows, cols)
    elif compression_type == "LZW + Masquage":
        lzw = LZWCoder()
        decompressed_data = lzw.decompress(compressed_data)
        qcoeffs_loaded = np.frombuffer(decompressed_data, dtype=np.int16).reshape(rows, cols)
    
    # Reconstruct signal
    coeffs_recon = dequantifier(qcoeffs_loaded, q=q)
    signal_recon = imdct(coeffs_recon, frame_size=frame_size_loaded)
    signal_recon = normaliser_signal(signal_recon)
    
    # Save reconstructed audio
    audio_recon = signal_to_audio(signal_recon, fs)
    audio_recon.export(reconstructed_file, format="wav")
    
    file_label.setText(f"Fichier compressé avec {compression_type}\nFichier Compressé: output.irm")
    
    # Clear previous plot
    for i in reversed(range(main_layout.count())):
        widget = main_layout.itemAt(i).widget()
        if isinstance(widget, FigureCanvas):
            main_layout.removeWidget(widget)
            widget.deleteLater()
    
    # Plot original vs reconstructed signal
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(signal[:1000], label="Original")
    ax.plot(signal_recon[:1000], label="Reconstruit", alpha=0.7)
    ax.legend()
    ax.set_title("Comparaison Original vs Reconstruit")
    ax.grid(True)
    
    canvas = FigureCanvas(fig)
    canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    main_layout.addWidget(canvas, stretch=1)

def signal_to_audio(signal, fs):
    signal = np.clip(signal, -1, 1)
    pcm = (signal * 32767).astype(np.int16)
    audio = AudioSegment(pcm.tobytes(), frame_rate=fs, sample_width=2, channels=1)
    return audio

def load_file():
    global audio, file_path
    file_path, _ = QFileDialog.getOpenFileName(None, "Ouvrir le fichier audio", "", "Audio Files (*.wav *.mp3 *.flac *.ogg)")
    
    if file_path:
        audio = AudioSegment.from_file(file_path)
        file_label.setText(f"Loaded: {os.path.basename(file_path)}")
    else:
        file_label.setText("Aucun fichier sélectionné")

def play_compressed_audio():
    if os.path.exists(reconstructed_file):
        audio_recon = AudioSegment.from_file(reconstructed_file)
        play(audio_recon)
    else:
        file_label.setText("Aucun fichier compressé disponible")

def show_size_difference():
    if os.path.exists(original_file) and os.path.exists(compressed_file):
        original_size = os.path.getsize(original_file) / 1024
        compressed_size = os.path.getsize(compressed_file) / 1024
        diff = original_size - compressed_size
        file_label.setText(f"Différence: {diff:.2f} Ko\nOriginal: {original_size:.2f} Ko\nCompressé: {compressed_size:.2f} Ko")
    else:
        file_label.setText("Fichiers nécessaires non disponibles")

def show_compression_percentage():
    if os.path.exists(original_file) and os.path.exists(compressed_file):
        original_size = os.path.getsize(original_file) / 1024
        compressed_size = os.path.getsize(compressed_file) / 1024
        if original_size > 0:
            percentage = (1 - compressed_size / original_size) * 100
            file_label.setText(f"Compression: {percentage:.2f}%")
        else:
            file_label.setText("Taille originale invalide")
    else:
        file_label.setText("Fichiers nécessaires non disponibles")

app = QApplication(sys.argv) if not QApplication.instance() else QApplication.instance()

window = QWidget()
window.setWindowTitle("Project TNIM - Audio Compression")
window.setGeometry(100, 100, 1000, 700)
window.setStyleSheet("""
    QWidget {
        background-color: #2E2E2E;
        color: #FFFFFF;
        font-family: Arial, sans-serif;
        font-size: 14px;
    }
    QLabel {
        font-size: 16px;
        margin-bottom: 10px;
        text-align: center;
    }
    QPushButton {
        background-color: transparent;
        color: #8EBCFF;
        border-radius: 12px;
        padding: 8px;
        font-size: 12px;
        min-width: 120px;
        margin: 5px;
    }
    QPushButton:hover {
        background-color: #8EBCFF;
        color: white;
    }
    QComboBox {
        background-color: #3E3E3E;
        color: #FFFFFF;
        padding: 8px;
        border-radius: 5px;
        min-width: 200px;
        margin: 5px;
    }
    QComboBox::drop-down {
        border: none;
    }
""")

# Main layout with plot at top and controls below
main_layout = QVBoxLayout()
window.setLayout(main_layout)

# File label
file_label = QLabel("Aucun fichier sélectionné")
main_layout.addWidget(file_label)

# First row of buttons
button_row1 = QHBoxLayout()
load_button = QPushButton("Sélectionner")
load_button.clicked.connect(load_file)
button_row1.addWidget(load_button)

compression_combo = QComboBox()
compression_combo.addItems([
    "LZW only",
    "Huffman only",
    "Huffman + LZW",
    "Huffman + LZW + Masquage",
    "Huffman + Masquage",
    "LZW + Masquage"
])
button_row1.addWidget(compression_combo)

compress_button = QPushButton("Compresser")
compress_button.clicked.connect(compress_audio)
button_row1.addWidget(compress_button)

main_layout.addLayout(button_row1)

# Second row of buttons
button_row2 = QHBoxLayout()
play_button = QPushButton("Écouter")
play_button.clicked.connect(play_compressed_audio)
button_row2.addWidget(play_button)

size_diff_button = QPushButton("Différence taille")
size_diff_button.clicked.connect(show_size_difference)
button_row2.addWidget(size_diff_button)

compression_percentage_button = QPushButton("% Compression")
compression_percentage_button.clicked.connect(show_compression_percentage)
button_row2.addWidget(compression_percentage_button)

main_layout.addLayout(button_row2)

# Add stretch to push buttons up
main_layout.addStretch(1)

window.show()

sys.exit(app.exec())