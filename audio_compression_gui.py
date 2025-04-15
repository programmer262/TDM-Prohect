import sys
import os
import numpy as np
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel
from pydub import AudioSegment
from pydub.playback import play
from scipy.signal import butter, filtfilt
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
from huffman import HuffmanCoder
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

def sauvegarder_compression(filename, qcoeffs, fs, frame_size):
    huffman = HuffmanCoder(precision=0)
    flat_qcoeffs = qcoeffs.flatten().tolist()
    compressed_data = huffman.compress(flat_qcoeffs)
    
    with open(filename, 'wb') as f:
        header = np.array([fs, frame_size, qcoeffs.shape[0], qcoeffs.shape[1]], dtype=np.int32)
        header.tofile(f)
        f.write(compressed_data)
    
    return compressed_data

def charger_compression(filename):
    with open(filename, 'rb') as f:
        header = np.fromfile(f, dtype=np.int32, count=4)
        fs, frame_size, rows, cols = header
        compressed_data = f.read()
    
    huffman = HuffmanCoder(precision=0)
    flat_qcoeffs = huffman.decompress(compressed_data)
    qcoeffs = np.array(flat_qcoeffs, dtype=np.int16).reshape(rows, cols)
    
    return qcoeffs, fs, frame_size

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
        file_label.setText(f"Loaded: {file_path}")
    else:
        file_label.setText("Aucun fichier sélectionné")

def compress_audio():
    global audio, file_path
    if not audio:
        file_label.setText("Aucun fichier audio chargé")
        return
    
    audio.export(original_file, format="wav")
    fs = audio.frame_rate
    signal = audio_to_signal(audio)
    signal = centrer_signal(signal)
    signal = normaliser_signal(signal)

    frame_size = 1024
    coeffs = mdct(signal, frame_size=frame_size)
    q = 0.02
    qcoeffs = quantifier(coeffs, q=q)
    
    sauvegarder_compression(compressed_file, qcoeffs, fs, frame_size)
    
    qcoeffs_loaded, fs_loaded, frame_size_loaded = charger_compression(compressed_file)
    coeffs_recon = dequantifier(qcoeffs_loaded, q=q)
    signal_recon = imdct(coeffs_recon, frame_size=frame_size_loaded)
    signal_recon = normaliser_signal(signal_recon)
    
    audio_recon = signal_to_audio(signal_recon, fs)
    audio_recon.export(reconstructed_file, format="wav")
    
    file_label.setText(f"Fichier compressé: {compressed_file}")

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
        file_label.setText(f"Différence de taille: {diff:.2f} Ko\nOriginal: {original_size:.2f} Ko\nCompressé: {compressed_size:.2f} Ko")
    else:
        file_label.setText("Fichiers nécessaires non disponibles")

def show_compression_percentage():
    if os.path.exists(original_file) and os.path.exists(compressed_file):
        original_size = os.path.getsize(original_file) / 1024
        compressed_size = os.path.getsize(compressed_file) / 1024
        if original_size > 0:
            percentage = (1 - compressed_size / original_size) * 100
            file_label.setText(f"Pourcentage de compression: {percentage:.2f}%")
        else:
            file_label.setText("Taille originale invalide")
    else:
        file_label.setText("Fichiers nécessaires non disponibles")

app = QApplication(sys.argv) if not QApplication.instance() else QApplication.instance()

window = QWidget()
window.setWindowTitle("Project TNIM - Audio Compression")
window.setGeometry(100, 100, 1000, 500)
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
        border-radius: 18px;
        padding: 20px;
        font-size: 14px;
        width: 60%;
        margin-top: 30px;
    }
    QPushButton:hover {
        background-color: #8EBCFF;
        color: white;
    }
""")

layout = QVBoxLayout()

file_label = QLabel("Aucun fichier sélectionné")
layout.addWidget(file_label)

load_button = QPushButton("Sélectionner le fichier audio")
load_button.clicked.connect(load_file)
layout.addWidget(load_button)

compress_button = QPushButton("Compresser le fichier audio")
compress_button.clicked.connect(compress_audio)
layout.addWidget(compress_button)

play_button = QPushButton("Écouter le fichier compressé")
play_button.clicked.connect(play_compressed_audio)
layout.addWidget(play_button)

size_diff_button = QPushButton("Différence de taille")
size_diff_button.clicked.connect(show_size_difference)
layout.addWidget(size_diff_button)

compression_percentage_button = QPushButton("Pourcentage de compression")
compression_percentage_button.clicked.connect(show_compression_percentage)
layout.addWidget(compression_percentage_button)

window.setLayout(layout)
window.show()

sys.exit(app.exec())