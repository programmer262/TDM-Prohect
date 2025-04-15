# import sys
# import os
# import numpy as np
# from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
#                               QPushButton, QFileDialog, QLabel, QComboBox, QSizePolicy, QFrame,
#                               QSlider, QProgressBar, QStyleFactory, QTabWidget, QGridLayout, QToolTip,
#                               QSpacerItem, QGraphicsDropShadowEffect, QScrollArea)
# from PySide6.QtCore import (QPropertyAnimation, QEasingCurve, Qt, QTimer, 
#                            QSize, QRect, QPoint, QSequentialAnimationGroup, QParallelAnimationGroup)
# from PySide6.QtGui import QColor, QIcon, QPixmap, QFont, QPalette, QLinearGradient, QGradient, QBrush
# from pydub import AudioSegment
# from pydub.playback import play
# from scipy.signal import butter, filtfilt
# from scipy.fftpack import dct, idct
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# import matplotlib.pyplot as plt
# from huffman import HuffmanCoder
# from LZW import LZWCoder
# from psychoacousticmodel import PsychoacousticModel
# import warnings
# warnings.filterwarnings('ignore')

# # Global variables
# audio = None
# file_path = ""
# compressed_file = "output.irm"
# original_file = "original.wav"
# reconstructed_file = "reconstructed.wav"
# quantization_factor = 0.02
# frame_size = 1024

# # UI Color Scheme
# COLORS = {
#     'primary_dark': '#121212',
#     'secondary_dark': '#1E1E2E',
#     'panel_bg': '#1A2333',
#     'accent': '#7289DA',
#     'accent_light': '#A4B8FF',
#     'accent_dark': '#5A6FB5',
#     'text_primary': '#E0E0E0',
#     'text_secondary': '#B0C4DE',
#     'success': '#4CAF50',
#     'warning': '#FFC107',
#     'error': '#FF5252',
#     'highlight': '#BB86FC',
# }

# # Drop Area Class
# class DropArea(QLabel):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setAlignment(Qt.AlignCenter)
#         self.setText("Drop audio file here")
#         self.setAcceptDrops(True)
#         self.setMinimumHeight(100)
#         self.setStyleSheet(f"""
#             border: 2px dashed {COLORS['accent']};
#             border-radius: 8px;
#             padding: 20px;
#             background: {COLORS['panel_bg']};
#             color: {COLORS['text_secondary']};
#             font-size: 15px;
#         """)
        
#     def dragEnterEvent(self, event):
#         if event.mimeData().hasUrls():
#             event.acceptProposedAction()
#             self.setStyleSheet(f"""
#                 border: 2px dashed {COLORS['highlight']};
#                 border-radius: 8px;
#                 padding: 20px;
#                 background: #2A2D3E;
#                 color: {COLORS['text_primary']};
#                 font-size: 15px;
#             """)
    
#     def dragLeaveEvent(self, event):
#         self.setStyleSheet(f"""
#             border: 2px dashed {COLORS['accent']};
#             border-radius: 8px;
#             padding: 20px;
#             background: {COLORS['panel_bg']};
#             color: {COLORS['text_secondary']};
#             font-size: 15px;
#         """)
        
#     def dropEvent(self, event):
#         global audio, file_path
#         if event.mimeData().hasUrls():
#             url = event.mimeData().urls()[0]
#             file_path = url.toLocalFile()
#             if file_path.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
#                 load_audio_file(file_path)
#             self.setStyleSheet(f"""
#                 border: 2px dashed {COLORS['accent']};
#                 border-radius: 8px;
#                 padding: 20px;
#                 background: {COLORS['panel_bg']};
#                 color: {COLORS['text_secondary']};
#                 font-size: 15px;
#             """)

# # Styled Components
# class StyledButton(QPushButton):
#     def __init__(self, text, icon_path=None, parent=None):
#         super().__init__(text, parent)
#         self.setFont(QFont("Roboto", 11))
#         self.setCursor(Qt.PointingHandCursor)
        
#         shadow = QGraphicsDropShadowEffect()
#         shadow.setBlurRadius(15)
#         shadow.setColor(QColor(0, 0, 0, 80))
#         shadow.setOffset(0, 2)
#         self.setGraphicsEffect(shadow)
        
#         if icon_path:
#             self.setIcon(QIcon(QPixmap(icon_path)))
#             self.setIconSize(QSize(18, 18))
        
#         self.setStyleSheet(f"""
#             QPushButton {{
#                 background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
#                                           stop:0 #2A3152, stop:1 #1E2A44);
#                 color: {COLORS['text_primary']};
#                 border: none;
#                 border-radius: 6px;
#                 padding: 10px 18px;
#                 font-weight: 500;
#             }}
#             QPushButton:hover {{
#                 background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
#                                           stop:0 {COLORS['accent']}, stop:1 {COLORS['accent_dark']});
#                 color: white;
#             }}
#             QPushButton:pressed {{
#                 background: {COLORS['accent_dark']};
#                 padding-top: 11px;
#                 padding-left: 19px;
#                 padding-bottom: 9px;
#                 padding-right: 17px;
#             }}
#             QPushButton:disabled {{
#                 background: #2A2A3A;
#                 color: #808080;
#             }}
#         """)

# class InfoCard(QFrame):
#     def __init__(self, title, parent=None):
#         super().__init__(parent)
#         self.setFrameShape(QFrame.StyledPanel)
        
#         shadow = QGraphicsDropShadowEffect()
#         shadow.setBlurRadius(15)
#         shadow.setColor(QColor(0, 0, 0, 60))
#         shadow.setOffset(0, 3)
#         self.setGraphicsEffect(shadow)
        
#         layout = QVBoxLayout()
#         layout.setContentsMargins(10, 10, 10, 10)
#         self.setLayout(layout)
        
#         title_label = QLabel(title)
#         title_label.setStyleSheet(f"""
#             font-size: 15px;
#             font-weight: bold;
#             color: {COLORS['accent']};
#             padding-bottom: 6px;
#             border-bottom: 1px solid {COLORS['accent_dark']};
#         """)
        
#         layout.addWidget(title_label)
        
#         self.content_layout = QVBoxLayout()
#         self.content_layout.setSpacing(5)
#         layout.addLayout(self.content_layout)
        
#         self.setStyleSheet(f"""
#             background: {COLORS['panel_bg']};
#             border-radius: 8px;
#         """)
    
#     def add_widget(self, widget):
#         self.content_layout.addWidget(widget)
    
#     def add_layout(self, layout):
#         self.content_layout.addLayout(layout)

# class EnhancedProgressBar(QProgressBar):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setRange(0, 100)
#         self.setValue(0)
#         self.setTextVisible(True)
#         self.setFormat("%p% - Ready")
#         self.setMinimumHeight(22)
#         self.setStyleSheet(f"""
#             QProgressBar {{
#                 border: 1px solid {COLORS['accent']};
#                 border-radius: 4px;
#                 text-align: center;
#                 background: {COLORS['panel_bg']};
#                 color: {COLORS['text_secondary']};
#                 font-size: 12px;
#                 font-weight: bold;
#             }}
#             QProgressBar::chunk {{
#                 background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
#                                                 stop:0 {COLORS['accent_dark']}, stop:1 {COLORS['accent']});
#                 border-radius: 3px;
#             }}
#         """)
    
#     def update_progress(self, value, message=""):
#         self.setValue(value)
#         if message:
#             self.setFormat(f"{value}% - {message}")
#         else:
#             self.setFormat(f"{value}%")

# # Animation Functions
# def pulse_effect(widget):
#     pulse_anim = QPropertyAnimation(widget, b"geometry")
#     pulse_anim.setDuration(200)
#     orig_geo = widget.geometry()
#     pulse_geo = QRect(orig_geo.x()-5, orig_geo.y()-5, 
#                       orig_geo.width()+10, orig_geo.height()+10)
#     pulse_anim.setStartValue(orig_geo)
#     pulse_anim.setEndValue(pulse_geo)
#     pulse_anim.setEasingCurve(QEasingCurve.OutQuad)
    
#     pulse_anim_back = QPropertyAnimation(widget, b"geometry")
#     pulse_anim_back.setDuration(200)
#     pulse_anim_back.setStartValue(pulse_geo)
#     pulse_anim_back.setEndValue(orig_geo)
#     pulse_anim_back.setEasingCurve(QEasingCurve.InQuad)
    
#     group = QSequentialAnimationGroup()
#     group.addAnimation(pulse_anim)
#     group.addAnimation(pulse_anim_back)
#     group.start()

# def highlight_widget(widget, duration=500):
#     original_style = widget.styleSheet()
#     highlight_style = f"""
#         border: 2px solid {COLORS['highlight']};
#         background-color: rgba(187, 134, 252, 0.15);
#         border-radius: 6px;
#     """
#     widget.setProperty("originalStyle", original_style)
#     merged_style = original_style + highlight_style
    
#     animation = QPropertyAnimation(widget, b"styleSheet")
#     animation.setDuration(duration)
#     animation.setStartValue(original_style)
#     animation.setEndValue(merged_style)
#     animation.setEasingCurve(QEasingCurve.OutCubic)
    
#     QTimer.singleShot(duration + 1000, lambda: widget.setStyleSheet(original_style))
#     animation.start()

# # Core Audio Functions
# def centrer_signal(signal):
#     return signal - np.mean(signal)

# def normaliser_signal(signal):
#     max_abs = np.max(np.abs(signal))
#     return signal / max_abs if max_abs != 0 else signal

# def audio_to_signal(audio):
#     signal = np.array(audio.get_array_of_samples()).astype(np.float32)
#     return signal / max(abs(signal))

# def mdct(signal, frame_size=1024):
#     N = frame_size
#     overlap = N // 2
#     nb_frames = (len(signal) - overlap) // overlap
#     mdct_coeffs = []
#     window = np.sin(np.pi / N * (np.arange(N) + 0.5))
#     for i in range(nb_frames):
#         frame = signal[i * overlap : i * overlap + N]
#         if len(frame) < N:
#             break
#         windowed = frame * window
#         coeffs = dct(windowed, type=2, norm='ortho')
#         mdct_coeffs.append(coeffs)
#     return np.array(mdct_coeffs)

# def imdct(coeffs, frame_size=1024):
#     N = frame_size
#     overlap = N // 2
#     window = np.sin(np.pi / N * (np.arange(N) + 0.5))
#     signal = np.zeros(overlap * (len(coeffs) + 1))
#     for i, c in enumerate(coeffs):
#         frame = idct(c, type=2, norm='ortho') * window
#         start = i * overlap
#         signal[start:start+N] += frame
#     return signal

# def quantifier_perceptual(coeffs, weights, base_q=0.02):
#     q_factors = base_q / weights
#     q_factors = np.clip(q_factors, base_q, base_q * 20)
#     qcoeffs = np.zeros_like(coeffs, dtype=np.int16)
#     for i in range(coeffs.shape[0]):
#         for j in range(coeffs.shape[1]):
#             qcoeffs[i, j] = int(round(coeffs[i, j] / q_factors[i, j]))
#     return qcoeffs, q_factors

# def dequantifier_perceptual(qcoeffs, q_factors):
#     coeffs_recon = np.zeros_like(qcoeffs, dtype=np.float32)
#     for i in range(qcoeffs.shape[0]):
#         for j in range(qcoeffs.shape[1]):
#             coeffs_recon[i, j] = qcoeffs[i, j] * q_factors[i, j]
#     return coeffs_recon

# def quantifier(coeffs, q=0.02):
#     return np.round(coeffs / q).astype(np.int16)

# def dequantifier(qcoeffs, q=0.02):
#     return qcoeffs.astype(np.float32) * q

# def signal_to_audio(signal, fs):
#     signal = np.clip(signal, -1, 1)
#     pcm = (signal * 32767).astype(np.int16)
#     audio = AudioSegment(pcm.tobytes(), frame_rate=fs, sample_width=2, channels=1)
#     return audio

# # Load and Compression Functions
# def load_audio_file(file_path):
#     global audio
#     try:
#         audio = AudioSegment.from_file(file_path)
#         file_label.setText(f"File loaded: {os.path.basename(file_path)}")
        
#         duration_label.setText(f"{len(audio)/1000:.2f} seconds")
#         channels_label.setText(f"{audio.channels}")
#         sample_rate_label.setText(f"{audio.frame_rate} Hz")
#         bit_depth_label.setText(f"{audio.sample_width * 8} bits")
        
#         file_name = os.path.basename(file_path)
#         if recent_files_combo.findText(file_name) == -1:
#             recent_files_combo.addItem(file_name)
        
#         compress_button.setEnabled(True)
#         play_original_button.setEnabled(True)
        
#         highlight_widget(file_info_card)
#         plot_original_waveform()
        
#         return True
#     except Exception as e:
#         file_label.setText(f"Error loading file: {str(e)}")
#         return False

# def compress_audio():
#     global audio, file_path, quantization_factor
#     if not audio:
#         file_label.setText("No audio file loaded")
#         return

#     compression_type = compression_combo.currentText()
#     progress_bar.show()
#     progress_bar.update_progress(0, "Starting compression...")

#     compress_button.setEnabled(False)
#     load_button.setEnabled(False)
    
#     QTimer.singleShot(100, lambda: process_compression(compression_type))

# def process_compression(compression_type):
#     global audio, file_path
#     audio.export(original_file, format="wav")
#     fs = audio.frame_rate
#     signal = audio_to_signal(audio)
#     signal = centrer_signal(signal)
#     signal = normaliser_signal(signal)
    
#     progress_bar.update_progress(10, "Transforming audio...")
    
#     coeffs = mdct(signal, frame_size=frame_size)
#     QTimer.singleShot(200, lambda: continue_compression(coeffs, fs, compression_type))

# def continue_compression(coeffs, fs, compression_type):
#     q = quantization_factor
#     qcoeffs = quantifier(coeffs, q=q)
    
#     progress_bar.update_progress(30, "Compressing data...")
    
#     if compression_type == "LZW only":
#         lzw = LZWCoder()
#         compressed_data = lzw.compress(qcoeffs.flatten().tobytes())
#     elif compression_type == "Huffman only":
#         huffman = HuffmanCoder(precision=0)
#         flat_qcoeffs = qcoeffs.flatten().tolist()
#         compressed_data = huffman.compress(flat_qcoeffs)
#     elif compression_type == "Huffman + LZW":
#         huffman = HuffmanCoder(precision=0)
#         flat_qcoeffs = qcoeffs.flatten().tolist()
#         huffman_compressed = huffman.compress(flat_qcoeffs)
#         lzw = LZWCoder()
#         compressed_data = lzw.compress(huffman_compressed)
#     elif compression_type == "Huffman + LZW + Masquage":
#         psycho_model = PsychoacousticModel(fs)
#         weights = psycho_model.perceptual_bit_allocation(coeffs, frame_size)
#         qcoeffs, q_factors = quantifier_perceptual(coeffs, weights, q)
#         huffman = HuffmanCoder(precision=0)
#         flat_qcoeffs = qcoeffs.flatten().tolist()
#         huffman_compressed = huffman.compress(flat_qcoeffs)
#         lzw = LZWCoder()
#         compressed_data = lzw.compress(huffman_compressed)
#     elif compression_type == "Huffman + Masquage":
#         psycho_model = PsychoacousticModel(fs)
#         weights = psycho_model.perceptual_bit_allocation(coeffs, frame_size)
#         qcoeffs, q_factors = quantifier_perceptual(coeffs, weights, q)
#         huffman = HuffmanCoder(precision=0)
#         flat_qcoeffs = qcoeffs.flatten().tolist()
#         compressed_data = huffman.compress(flat_qcoeffs)
#     elif compression_type == "LZW + Masquage":
#         psycho_model = PsychoacousticModel(fs)
#         weights = psycho_model.perceptual_bit_allocation(coeffs, frame_size)
#         qcoeffs, q_factors = quantifier_perceptual(coeffs, weights, q)
#         lzw = LZWCoder()
#         compressed_data = lzw.compress(qcoeffs.flatten().tobytes())
    
#     with open(compressed_file, 'wb') as f:
#         header = np.array([fs, frame_size, qcoeffs.shape[0], qcoeffs.shape[1]], dtype=np.int32)
#         header.tofile(f)
#         f.write(compressed_data)
    
#     progress_bar.update_progress(50, "Decompressing data...")
    
#     with open(compressed_file, 'rb') as f:
#         header = np.fromfile(f, dtype=np.int32, count=4)
#         fs_loaded, frame_size_loaded, rows, cols = header
#         compressed_data = f.read()
    
#     QTimer.singleShot(300, lambda: decompress_data(compressed_data, compression_type, fs_loaded, frame_size_loaded, rows, cols, q))

# def decompress_data(compressed_data, compression_type, fs_loaded, frame_size_loaded, rows, cols, q):
#     if compression_type == "LZW only":
#         lzw = LZWCoder()
#         decompressed_data = lzw.decompress(compressed_data)
#         qcoeffs_loaded = np.frombuffer(decompressed_data, dtype=np.int16).reshape(rows, cols)
#         coeffs_recon = dequantifier(qcoeffs_loaded, q=q)
#     elif compression_type == "Huffman only":
#         huffman = HuffmanCoder(precision=0)
#         flat_qcoeffs = huffman.decompress(compressed_data)
#         qcoeffs_loaded = np.array(flat_qcoeffs, dtype=np.int16).reshape(rows, cols)
#         coeffs_recon = dequantifier(qcoeffs_loaded, q=q)
#     elif compression_type == "Huffman + LZW":
#         lzw = LZWCoder()
#         huffman_data = lzw.decompress(compressed_data)
#         huffman = HuffmanCoder(precision=0)
#         flat_qcoeffs = huffman.decompress(huffman_data)
#         qcoeffs_loaded = np.array(flat_qcoeffs, dtype=np.int16).reshape(rows, cols)
#         coeffs_recon = dequantifier(qcoeffs_loaded, q=q)
#     elif compression_type in ["Huffman + LZW + Masquage", "Huffman + Masquage", "LZW + Masquage"]:
#         if compression_type == "Huffman + LZW + Masquage":
#             lzw = LZWCoder()
#             huffman_data = lzw.decompress(compressed_data)
#             huffman = HuffmanCoder(precision=0)
#             flat_qcoeffs = huffman.decompress(huffman_data)
#             qcoeffs_loaded = np.array(flat_qcoeffs, dtype=np.int16).reshape(rows, cols)
#         elif compression_type == "Huffman + Masquage":
#             huffman = HuffmanCoder(precision=0)
#             flat_qcoeffs = huffman.decompress(compressed_data)
#             qcoeffs_loaded = np.array(flat_qcoeffs, dtype=np.int16).reshape(rows, cols)
#         else:
#             lzw = LZWCoder()
#             decompressed_data = lzw.decompress(compressed_data)
#             qcoeffs_loaded = np.frombuffer(decompressed_data, dtype=np.int16).reshape(rows, cols)
#         coeffs_recon = dequantifier(qcoeffs_loaded, q=q)
    
#     progress_bar.update_progress(80, "Reconstructing audio...")
    
#     signal_recon = imdct(coeffs_recon, frame_size=frame_size_loaded)
#     signal_recon = normaliser_signal(signal_recon)
    
#     audio_recon = signal_to_audio(signal_recon, fs_loaded)
#     audio_recon.export(reconstructed_file, format="wav")
    
#     QTimer.singleShot(400, lambda: finish_compression(compression_type))

# def finish_compression(compression_type):
#     update_stats()
#     plot_comparison()
    
#     compress_button.setEnabled(True)
#     load_button.setEnabled(True)
#     play_compressed_button.setEnabled(True)
    
#     pulse_effect(stats_frame)
#     file_label.setText(f"Compressed with {compression_type}\nOutput: output.irm")
#     progress_bar.update_progress(100, "Compression completed!")
    
#     QTimer.singleShot(2000, lambda: progress_bar.hide())

# def play_compressed_audio():
#     if os.path.exists(reconstructed_file):
#         audio_recon = AudioSegment.from_file(reconstructed_file)
#         play(audio_recon)
#         highlight_widget(play_compressed_button)
#     else:
#         file_label.setText("No compressed file available")

# def play_original_audio():
#     if audio:
#         play(audio)
#         highlight_widget(play_original_button)
#     else:
#         file_label.setText("No audio file loaded")

# def show_size_difference():
#     if os.path.exists(original_file) and os.path.exists(compressed_file):
#         original_size = os.path.getsize(original_file) / 1024
#         compressed_size = os.path.getsize(compressed_file) / 1024
#         diff = original_size - compressed_size
#         stats_labels["difference"].setText(f"{diff:.2f} KB")
#         highlight_widget(stats_frame)
#     else:
#         file_label.setText("Required files not available")

# def show_compression_percentage():
#     if os.path.exists(original_file) and os.path.exists(compressed_file):
#         original_size = os.path.getsize(original_file) / 1024
#         compressed_size = os.path.getsize(compressed_file) / 1024
#         if original_size > 0:
#             percentage = (1 - compressed_size / original_size) * 100
#             stats_labels["percentage"].setText(f"{percentage:.2f}%")
#             highlight_widget(stats_frame)
#         else:
#             file_label.setText("Invalid original size")
#     else:
#         file_label.setText("Required files not available")

# def update_stats():
#     if os.path.exists(original_file) and os.path.exists(compressed_file):
#         original_size = os.path.getsize(original_file) / 1024
#         compressed_size = os.path.getsize(compressed_file) / 1024
#         diff = original_size - compressed_size
#         percentage = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
#         stats_labels["original"].setText(f"{original_size:.2f} KB")
#         stats_labels["compressed"].setText(f"{compressed_size:.2f} KB")
#         stats_labels["difference"].setText(f"{diff:.2f} KB")
#         stats_labels["percentage"].setText(f"{percentage:.2f}%")

# def plot_original_waveform():
#     if audio:
#         signal = normaliser_signal(audio_to_signal(audio))
        
#         for i in reversed(range(waveform_preview_layout.count())):
#             widget = waveform_preview_layout.itemAt(i).widget()
#             if widget:
#                 waveform_preview_layout.removeWidget(widget)
#                 widget.deleteLater()
        
#         fig, ax = plt.subplots(figsize=(6, 2))
#         plt.style.use('dark_background')
#         ax.plot(signal[:5000], color=COLORS['accent_light'], linewidth=0.8)
#         ax.set_title("Audio Waveform Preview", color=COLORS['text_primary'])
#         ax.set_facecolor(COLORS['panel_bg'])
#         fig.patch.set_facecolor(COLORS['panel_bg'])
#         ax.grid(True, alpha=0.3)
#         ax.set_xlim(0, 5000)
#         ax.set_ylim(-1, 1)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['bottom'].set_color(COLORS['accent_dark'])
#         ax.spines['left'].set_color(COLORS['accent_dark'])
#         ax.tick_params(axis='x', colors=COLORS['text_secondary'])
#         ax.tick_params(axis='y', colors=COLORS['text_secondary'])
        
#         canvas = FigureCanvas(fig)
#         waveform_preview_layout.addWidget(canvas)

# def plot_comparison():
#     global visualization_layout
#     for i in reversed(range(visualization_layout.count())):
#         widget = visualization_layout.itemAt(i).widget()
#         if widget:
#             visualization_layout.removeWidget(widget)
#             widget.deleteLater()

#     signal = normaliser_signal(audio_to_signal(audio))
#     signal_recon = normaliser_signal(audio_to_signal(AudioSegment.from_file(reconstructed_file)))

#     fig1, ax1 = plt.subplots(figsize=(6, 2))
#     plt.style.use('dark_background')
#     ax1.plot(signal[:1000], label="Original", color=COLORS['accent_light'])
#     ax1.set_title("Original Signal", color=COLORS['text_primary'])
#     ax1.set_facecolor(COLORS['panel_bg'])
#     fig1.patch.set_facecolor(COLORS['panel_bg'])
#     ax1.grid(True, alpha=0.3)
#     ax1.spines['top'].set_visible(False)
#     ax1.spines['right'].set_visible(False)

#     fig2, ax2 = plt.subplots(figsize=(6, 2))
#     plt.style.use('dark_background')
#     ax2.plot(signal_recon[:1000], label="Reconstructed", color=COLORS['highlight'])
#     ax2.set_title("Reconstructed Signal", color=COLORS['text_primary'])
#     ax2.set_facecolor(COLORS['panel_bg'])
#     fig2.patch.set_facecolor(COLORS['panel_bg'])
#     ax2.grid(True, alpha=0.3)
#     ax2.spines['top'].set_visible(False)
#     ax2.spines['right'].set_visible(False)

#     canvas1 = FigureCanvas(fig1)
#     canvas2 = FigureCanvas(fig2)
#     visualization_layout.addWidget(canvas1, 0, 0)
#     visualization_layout.addWidget(canvas2, 0, 1)

# # Main Application Setup
# app = QApplication(sys.argv) if not QApplication.instance() else QApplication.instance()
# app.setStyle(QStyleFactory.create("Fusion"))
# app.setFont(QFont("Roboto", 11))

# dark_palette = QPalette()
# dark_palette.setColor(QPalette.Window, QColor(COLORS['primary_dark']))
# dark_palette.setColor(QPalette.WindowText, QColor(COLORS['text_primary']))
# dark_palette.setColor(QPalette.Base, QColor(COLORS['secondary_dark']))
# dark_palette.setColor(QPalette.AlternateBase, QColor(COLORS['panel_bg']))
# dark_palette.setColor(QPalette.ToolTipBase, QColor(COLORS['accent']))
# dark_palette.setColor(QPalette.ToolTipText, QColor(COLORS['text_primary']))
# dark_palette.setColor(QPalette.Text, QColor(COLORS['text_primary']))
# dark_palette.setColor(QPalette.Button, QColor(COLORS['panel_bg']))
# dark_palette.setColor(QPalette.ButtonText, QColor(COLORS['text_primary']))
# dark_palette.setColor(QPalette.Link, QColor(COLORS['accent']))
# dark_palette.setColor(QPalette.Highlight, QColor(COLORS['accent']))
# dark_palette.setColor(QPalette.HighlightedText, QColor(COLORS['text_primary']))
# app.setPalette(dark_palette)

# QToolTip.setFont(QFont('Roboto', 11))
# app.setStyleSheet(f"""
#     QToolTip {{
#         background-color: {COLORS['panel_bg']};
#         color: {COLORS['text_primary']};
#         border: 1px solid {COLORS['accent']};
#         border-radius: 4px;
#         padding: 5px;
#     }}
# """)

# window = QWidget()
# window.setWindowTitle("IRM Audio compressor")
# window.setGeometry(100, 100, 1400, 900)
# window.setMinimumSize(1000, 700)

# main_layout = QVBoxLayout()
# main_layout.setContentsMargins(15, 15, 15, 15)
# window.setLayout(main_layout)

# header_frame = QFrame()
# header_layout = QHBoxLayout()
# header_layout.setContentsMargins(0, 0, 0, 8)
# header_frame.setLayout(header_layout)

# logo_label = QLabel()
# logo_pixmap = QPixmap("icons/logo.png")
# if not logo_pixmap.isNull():
#     logo_label.setPixmap(logo_pixmap.scaled(35, 35, Qt.KeepAspectRatio, Qt.SmoothTransformation))
# else:
#     logo_label.setText("🎵")
#     logo_label.setStyleSheet(f"font-size: 28px; color: {COLORS['accent']};")
# header_layout.addWidget(logo_label)

# header_label = QLabel("IRM Audio Compressor")
# header_label.setStyleSheet(f"""
#     font-size: 24px; 
#     font-weight: bold; 
#     color: {COLORS['accent']}; 
#     padding-left: 8px;
# """)
# header_layout.addWidget(header_label)

# version_label = QLabel("v2.0")
# version_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px; padding-left: 8px;")
# header_layout.addWidget(version_label)

# header_layout.addStretch()

# help_button = StyledButton("?", None)
# help_button.setFixedSize(28, 28)
# help_button.setStyleSheet(f"""
#     QPushButton {{
#         background: {COLORS['accent']};
#         color: white;
#         border-radius: 14px;
#         font-weight: bold;
#         font-size: 16px;
#     }}
#     QPushButton:hover {{
#         background: {COLORS['accent_light']};
#     }}
# """)
# help_button.setToolTip("Show help and documentation")
# header_layout.addWidget(help_button)

# main_layout.addWidget(header_frame)

# tabs = QTabWidget()
# tabs.setStyleSheet(f"""
#     QTabWidget::pane {{
#         border: 1px solid {COLORS['accent']};
#         border-radius: 6px;
#         margin-top: -1px;
#     }}
#     QTabBar::tab {{
#         background: {COLORS['panel_bg']};
#         color: {COLORS['text_secondary']};
#         padding: 8px 18px;
#         margin: 2px;
#         border-top-left-radius: 6px;
#         border-top-right-radius: 6px;
#         font-size: 13px;
#     }}
#     QTabBar::tab:selected {{
#         background: {COLORS['accent']};
#         color: white;
#         font-weight: bold;
#     }}
#     QTabBar::tab:hover:!selected {{
#         background: {COLORS['secondary_dark']};
#         color: {COLORS['text_primary']};
#     }}
# """)
# main_layout.addWidget(tabs)

# # Compression Tab
# compression_tab = QWidget()
# compression_layout = QHBoxLayout()
# compression_layout.setSpacing(15)
# compression_tab.setLayout(compression_layout)

# left_panel = QScrollArea()
# left_panel.setWidgetResizable(True)
# left_panel.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
# left_panel.setStyleSheet(f"""
#     QScrollArea {{
#         background: transparent;
#         border: none;
#     }}
#     QScrollBar:vertical {{
#         background: {COLORS['panel_bg']};
#         width: 8px;
#         margin: 0px;
#     }}
#     QScrollBar::handle:vertical {{
#         background: {COLORS['accent_dark']};
#         min-height: 20px;
#         border-radius: 4px;
#     }}
#     QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
#         height: 0px;
#     }}
# """)
# left_panel.setFixedWidth(320)

# left_widget = QWidget()
# left_layout = QVBoxLayout()
# left_layout.setSpacing(10)
# left_widget.setLayout(left_layout)
# left_panel.setWidget(left_widget)

# # File Info Section
# file_info_card = InfoCard("Audio File Information")

# file_label = QLabel("No file selected")
# file_label.setStyleSheet(f"""
#     font-size: 13px; 
#     padding: 8px; 
#     background: {COLORS['secondary_dark']}; 
#     border-radius: 4px; 
#     color: {COLORS['text_secondary']};
# """)
# file_label.setAlignment(Qt.AlignCenter)
# file_info_card.add_widget(file_label)

# file_details_layout = QGridLayout()
# file_details_layout.setColumnStretch(1, 1)
# file_details_layout.setVerticalSpacing(5)

# file_details_layout.addWidget(QLabel("Duration:"), 0, 0)
# duration_label = QLabel("0.00 seconds")
# duration_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
# file_details_layout.addWidget(duration_label, 0, 1)

# file_details_layout.addWidget(QLabel("Channels:"), 1, 0)
# channels_label = QLabel("0")
# channels_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
# file_details_layout.addWidget(channels_label, 1, 1)

# file_details_layout.addWidget(QLabel("Sample Rate:"), 2, 0)
# sample_rate_label = QLabel("0 Hz")
# sample_rate_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
# file_details_layout.addWidget(sample_rate_label, 2, 1)

# file_details_layout.addWidget(QLabel("Bit Depth:"), 3, 0)
# bit_depth_label = QLabel("0 bits")
# bit_depth_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
# file_details_layout.addWidget(bit_depth_label, 3, 1)

# file_info_card.add_layout(file_details_layout)
# left_layout.addWidget(file_info_card)

# # Controls Section
# controls_card = InfoCard("Compression Controls")

# drop_area = DropArea()
# controls_card.add_widget(drop_area)

# load_button = StyledButton("Select File", "icons/load.png")
# load_button.clicked.connect(lambda: load_audio_file(QFileDialog.getOpenFileName(None, "Open Audio File", "", "Audio Files (*.wav *.mp3 *.flac *.ogg)")[0]))
# controls_card.add_widget(load_button)

# recent_files_label = QLabel("Recent Files:")
# recent_files_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
# controls_card.add_widget(recent_files_label)

# recent_files_combo = QComboBox()
# recent_files_combo.setStyleSheet(f"""
#     QComboBox {{
#         background: {COLORS['secondary_dark']};
#         color: {COLORS['text_primary']};
#         padding: 8px;
#         border-radius: 4px;
#         font-size: 12px;
#     }}
#     QComboBox::drop-down {{
#         border: none;
#         width: 25px;
#     }}
#     QComboBox QAbstractItemView {{
#         background: {COLORS['secondary_dark']};
#         color: {COLORS['text_primary']};
#         selection-background-color: {COLORS['accent']};
#     }}
# """)
# recent_files_combo.setToolTip("Select a recently used file")
# controls_card.add_widget(recent_files_combo)

# compression_label = QLabel("Compression Method:")
# compression_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px; margin-top: 10px;")
# controls_card.add_widget(compression_label)

# compression_combo = QComboBox()
# compression_combo.addItems([
#     "LZW only",
#     "Huffman only",
#     "Huffman + LZW",
#     "Huffman + LZW + Masquage",
#     "Huffman + Masquage",
#     "LZW + Masquage"
# ])
# compression_combo.setStyleSheet(f"""
#     QComboBox {{
#         background: {COLORS['secondary_dark']};
#         color: {COLORS['text_primary']};
#         padding: 8px;
#         border-radius: 4px;
#         font-size: 12px;
#     }}
#     QComboBox::drop-down {{
#         border: none;
#         width: 25px;
#     }}
#     QComboBox QAbstractItemView {{
#         background: {COLORS['secondary_dark']};
#         color: {COLORS['text_primary']};
#         selection-background-color: {COLORS['accent']};
#     }}
# """)
# compression_combo.setToolTip("Choose a compression method")
# controls_card.add_widget(compression_combo)

# compress_button = StyledButton("Compress", "icons/compress.png")
# compress_button.clicked.connect(compress_audio)
# compress_button.setEnabled(False)
# controls_card.add_widget(compress_button)

# playback_layout = QHBoxLayout()
# playback_layout.setSpacing(10)

# play_original_button = StyledButton("Play Original", "icons/play.png")
# play_original_button.clicked.connect(play_original_audio)
# play_original_button.setEnabled(False)
# playback_layout.addWidget(play_original_button)

# play_compressed_button = StyledButton("Play Compressed", "icons/play.png")
# play_compressed_button.clicked.connect(play_compressed_audio)
# play_compressed_button.setEnabled(False)
# playback_layout.addWidget(play_compressed_button)

# controls_card.add_layout(playback_layout)

# progress_bar = EnhancedProgressBar()
# progress_bar.hide()
# controls_card.add_widget(progress_bar)

# left_layout.addWidget(controls_card)

# # Stats Section
# stats_card = InfoCard("Compression Statistics")
# stats_frame = QFrame()
# stats_layout = QGridLayout()
# stats_layout.setVerticalSpacing(5)
# stats_frame.setLayout(stats_layout)

# stats_labels = {}
# labels = ["Original Size:", "Compressed Size:", "Difference:", "Compression Rate:"]
# keys = ["original", "compressed", "difference", "percentage"]
# for i, (label, key) in enumerate(zip(labels, keys)):
#     lbl = QLabel(label)
#     lbl.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
#     stats_layout.addWidget(lbl, i, 0)
#     stats_labels[key] = QLabel("0.00")
#     stats_labels[key].setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 12px;")
#     stats_layout.addWidget(stats_labels[key], i, 1)

# stats_card.add_widget(stats_frame)

# stats_buttons_layout = QHBoxLayout()
# stats_buttons_layout.setSpacing(10)

# size_diff_button = StyledButton("Show Difference", "icons/size.png")
# size_diff_button.clicked.connect(show_size_difference)
# stats_buttons_layout.addWidget(size_diff_button)

# percentage_button = StyledButton("Show %", "icons/percentage.png")
# percentage_button.clicked.connect(show_compression_percentage)
# stats_buttons_layout.addWidget(percentage_button)

# stats_card.add_layout(stats_buttons_layout)
# left_layout.addWidget(stats_card)

# left_layout.addStretch()
# compression_layout.addWidget(left_panel)

# # Right Panel (Visualization)
# right_panel = QFrame()
# right_layout = QVBoxLayout()
# right_layout.setSpacing(10)
# right_panel.setLayout(right_layout)

# visualization_label = QLabel("Visualization")
# visualization_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {COLORS['accent']}; margin: 5px;")
# visualization_label.setAlignment(Qt.AlignCenter)
# right_layout.addWidget(visualization_label)

# waveform_card = InfoCard("Waveform Preview")
# waveform_preview_layout = QVBoxLayout()
# waveform_card.add_layout(waveform_preview_layout)
# right_layout.addWidget(waveform_card)

# visualization_card = InfoCard("Compression Analysis")
# visualization_layout = QGridLayout()
# visualization_layout.setSpacing(10)
# visualization_card.add_layout(visualization_layout)
# right_layout.addWidget(visualization_card)

# compression_layout.addWidget(right_panel)
# tabs.addTab(compression_tab, "Compression")

# # # Settings Tab (Placeholder)
# # settings_tab = QWidget()
# # settings_layout = QVBoxLayout()
# # settings_tab.setLayout(settings_layout)
# # settings_label = QLabel("Advanced Settings (Coming Soon)")
# # settings_label.setAlignment(Qt.AlignCenter)
# # settings_layout.addWidget(settings_label)
# # tabs.addTab(settings_tab, "Settings")

# window.show()
# sys.exit(app.exec())




import sys
import os
import numpy as np
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                              QPushButton, QFileDialog, QLabel, QComboBox, QSizePolicy, QFrame,
                              QSlider, QProgressBar, QStyleFactory, QTabWidget, QGridLayout, QToolTip,
                              QSpacerItem, QGraphicsDropShadowEffect, QScrollArea, QDialog, QTextBrowser)
from PySide6.QtCore import (QPropertyAnimation, QEasingCurve, Qt, QTimer, 
                           QSize, QRect, QPoint, QSequentialAnimationGroup, QParallelAnimationGroup)
from PySide6.QtGui import QColor, QIcon, QPixmap, QFont, QPalette, QLinearGradient, QGradient, QBrush
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
quantization_factor = 0.02
frame_size = 1024

# UI Color Scheme
COLORS = {
    'primary_dark': '#121212',
    'secondary_dark': '#1E1E2E',
    'panel_bg': '#1A2333',
    'accent': '#7289DA',
    'accent_light': '#A4B8FF',
    'accent_dark': '#5A6FB5',
    'text_primary': '#E0E0E0',
    'text_secondary': '#B0C4DE',
    'success': '#4CAF50',
    'warning': '#FFC107',
    'error': '#FF5252',
    'highlight': '#BB86FC',
}

# Help Dialog Class
class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Help - Audio Compressor Elite")
        self.setFixedSize(500, 400)
        self.setStyleSheet(f"""
            background: {COLORS['panel_bg']};
            color: {COLORS['text_primary']};
            border-radius: 8px;
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        self.setLayout(layout)
        
        help_text = QTextBrowser()
        help_text.setStyleSheet(f"""
            background: {COLORS['secondary_dark']};
            color: {COLORS['text_primary']};
            border: 1px solid {COLORS['accent_dark']};
            border-radius: 4px;
            padding: 10px;
            font-size: 13px;
        """)
        help_text.setOpenExternalLinks(True)
        help_text.setHtml("""
            <h2 style='color: #7289DA;'>Welcome to Audio Compressor Elite</h2>
            <p><b>Purpose:</b> This GUI application enables compression and decompression of audio files using advanced algorithms to reduce file size while maintaining quality.</p>
            <p><b>Functionality:</b></p>
            <ul>
                <li>Load audio files (WAV, MP3, FLAC, OGG) via drag-and-drop or file selection.</li>
                <li>Apply compression using various methods.</li>
                <li>Visualize original and compressed waveforms.</li>
                <li>Play original and compressed audio for comparison.</li>
                <li>View compression statistics (size reduction, percentage).</li>
            </ul>
            <p><b>Algorithms Used:</b></p>
            <ul>
                <li><b>Huffman Coding:</b> A lossless compression method that assigns variable-length codes to data based on frequency, reducing redundancy.</li>
                <li><b>LZW Compression:</b> A dictionary-based algorithm that builds patterns to replace repeated data with shorter codes, effective for repetitive sequences.</li>
            </ul>
            <p>Select a compression method from the dropdown to combine these algorithms with psychoacoustic modeling for optimal results.</p>
        """)
        
        layout.addWidget(help_text)
        
        close_button = QPushButton("Close")
        close_button.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['accent']};
                color: white;
                border-radius: 4px;
                padding: 8px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background: {COLORS['accent_light']};
            }}
        """)
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

# Drop Area Class
class DropArea(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Drop audio file here")
        self.setAcceptDrops(True)
        self.setMinimumHeight(100)
        self.setStyleSheet(f"""
            border: 2px dashed {COLORS['accent']};
            border-radius: 8px;
            padding: 20px;
            background: {COLORS['panel_bg']};
            color: {COLORS['text_secondary']};
            font-size: 15px;
        """)
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(f"""
                border: 2px dashed {COLORS['highlight']};
                border-radius: 8px;
                padding: 20px;
                background: #2A2D3E;
                color: {COLORS['text_primary']};
                font-size: 15px;
            """)
    
    def dragLeaveEvent(self, event):
        self.setStyleSheet(f"""
            border: 2px dashed {COLORS['accent']};
            border-radius: 8px;
            padding: 20px;
            background: {COLORS['panel_bg']};
            color: {COLORS['text_secondary']};
            font-size: 15px;
        """)
        
    def dropEvent(self, event):
        global audio, file_path
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                load_audio_file(file_path)
            self.setStyleSheet(f"""
                border: 2px dashed {COLORS['accent']};
                border-radius: 8px;
                padding: 20px;
                background: {COLORS['panel_bg']};
                color: {COLORS['text_secondary']};
                font-size: 15px;
            """)

# Styled Components
class StyledButton(QPushButton):
    def __init__(self, text, icon_path=None, parent=None):
        super().__init__(text, parent)
        self.setFont(QFont("Roboto", 11))
        self.setCursor(Qt.PointingHandCursor)
        
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        
        if icon_path:
            self.setIcon(QIcon(QPixmap(icon_path)))
            self.setIconSize(QSize(18, 18))
        
        self.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                          stop:0 #2A3152, stop:1 #1E2A44);
                color: {COLORS['text_primary']};
                border: none;
                border-radius: 6px;
                padding: 10px 18px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                          stop:0 {COLORS['accent']}, stop:1 {COLORS['accent_dark']});
                color: white;
            }}
            QPushButton:pressed {{
                background: {COLORS['accent_dark']};
                padding-top: 11px;
                padding-left: 19px;
                padding-bottom: 9px;
                padding-right: 17px;
            }}
            QPushButton:disabled {{
                background: #2A2A3A;
                color: #808080;
            }}
        """)

class InfoCard(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 60))
        shadow.setOffset(0, 3)
        self.setGraphicsEffect(shadow)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(layout)
        
        title_label = QLabel(title)
        title_label.setStyleSheet(f"""
            font-size: 15px;
            font-weight: bold;
            color: {COLORS['accent']};
            padding-bottom: 6px;
            border-bottom: 1px solid {COLORS['accent_dark']};
        """)
        
        layout.addWidget(title_label)
        
        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(5)
        layout.addLayout(self.content_layout)
        
        self.setStyleSheet(f"""
            background: {COLORS['panel_bg']};
            border-radius: 8px;
        """)
    
    def add_widget(self, widget):
        self.content_layout.addWidget(widget)
    
    def add_layout(self, layout):
        self.content_layout.addLayout(layout)

class EnhancedProgressBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(0, 100)
        self.setValue(0)
        self.setTextVisible(True)
        self.setFormat("%p% - Ready")
        self.setMinimumHeight(22)
        self.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {COLORS['accent']};
                border-radius: 4px;
                text-align: center;
                background: {COLORS['panel_bg']};
                color: {COLORS['text_secondary']};
                font-size: 12px;
                font-weight: bold;
            }}
            QProgressBar::chunk {{
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                                stop:0 {COLORS['accent_dark']}, stop:1 {COLORS['accent']});
                border-radius: 3px;
            }}
        """)
    
    def update_progress(self, value, message=""):
        self.setValue(value)
        if message:
            self.setFormat(f"{value}% - {message}")
        else:
            self.setFormat(f"{value}%")

# Animation Functions
def pulse_effect(widget):
    pulse_anim = QPropertyAnimation(widget, b"geometry")
    pulse_anim.setDuration(200)
    orig_geo = widget.geometry()
    pulse_geo = QRect(orig_geo.x()-5, orig_geo.y()-5,orig_geo.width()+10, orig_geo.height()+10)
    pulse_anim.setStartValue(orig_geo)
    pulse_anim.setEndValue(pulse_geo)
    pulse_anim.setEasingCurve(QEasingCurve.OutQuad)
    
    pulse_anim_back = QPropertyAnimation(widget, b"geometry")
    pulse_anim_back.setDuration(200)
    pulse_anim_back.setStartValue(pulse_geo)
    pulse_anim_back.setEndValue(orig_geo)
    pulse_anim_back.setEasingCurve(QEasingCurve.InQuad)
    
    group = QSequentialAnimationGroup()
    group.addAnimation(pulse_anim)
    group.addAnimation(pulse_anim_back)
    group.start()

def highlight_widget(widget, duration=500):
    original_style = widget.styleSheet()
    highlight_style = f"""
        border: 2px solid {COLORS['highlight']};
        background-color: rgba(187, 134, 252, 0.15);
        border-radius: 6px;
    """
    widget.setProperty("originalStyle", original_style)
    merged_style = original_style + highlight_style
    
    animation = QPropertyAnimation(widget, b"styleSheet")
    animation.setDuration(duration)
    animation.setStartValue(original_style)
    animation.setEndValue(merged_style)
    animation.setEasingCurve(QEasingCurve.OutCubic)
    
    QTimer.singleShot(duration + 1000, lambda: widget.setStyleSheet(original_style))
    animation.start()

# Core Audio Functions
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

def quantifier_perceptual(coeffs, weights, base_q=0.02):
    q_factors = base_q / weights
    q_factors = np.clip(q_factors, base_q, base_q * 20)
    qcoeffs = np.zeros_like(coeffs, dtype=np.int16)
    for i in range(coeffs.shape[0]):
        for j in range(coeffs.shape[1]):
            qcoeffs[i, j] = int(round(coeffs[i, j] / q_factors[i, j]))
    return qcoeffs, q_factors

def dequantifier_perceptual(qcoeffs, q_factors):
    coeffs_recon = np.zeros_like(qcoeffs, dtype=np.float32)
    for i in range(qcoeffs.shape[0]):
        for j in range(qcoeffs.shape[1]):
            coeffs_recon[i, j] = qcoeffs[i, j] * q_factors[i, j]
    return coeffs_recon

def quantifier(coeffs, q=0.02):
    return np.round(coeffs / q).astype(np.int16)

def dequantifier(qcoeffs, q=0.02):
    return qcoeffs.astype(np.float32) * q

def signal_to_audio(signal, fs):
    signal = np.clip(signal, -1, 1)
    pcm = (signal * 32767).astype(np.int16)
    audio = AudioSegment(pcm.tobytes(), frame_rate=fs, sample_width=2, channels=1)
    return audio

# Load and Compression Functions
def load_audio_file(file_path):
    global audio
    try:
        audio = AudioSegment.from_file(file_path)
        file_label.setText(f"File loaded: {os.path.basename(file_path)}")
        
        duration_label.setText(f"{len(audio)/1000:.2f} seconds")
        channels_label.setText(f"{audio.channels}")
        sample_rate_label.setText(f"{audio.frame_rate} Hz")
        bit_depth_label.setText(f"{audio.sample_width * 8} bits")
        
        file_name = os.path.basename(file_path)
        if recent_files_combo.findText(file_name) == -1:
            recent_files_combo.addItem(file_name)
        
        compress_button.setEnabled(True)
        play_original_button.setEnabled(True)
        
        highlight_widget(file_info_card)
        plot_original_waveform()
        
        return True
    except Exception as e:
        file_label.setText(f"Error loading file: {str(e)}")
        return False

def compress_audio():
    global audio, file_path, quantization_factor
    if not audio:
        file_label.setText("No audio file loaded")
        return

    compression_type = compression_combo.currentText()
    progress_bar.show()
    progress_bar.update_progress(0, "Starting compression...")

    compress_button.setEnabled(False)
    load_button.setEnabled(False)
    
    QTimer.singleShot(100, lambda: process_compression(compression_type))

def process_compression(compression_type):
    global audio, file_path
    audio.export(original_file, format="wav")
    fs = audio.frame_rate
    signal = audio_to_signal(audio)
    signal = centrer_signal(signal)
    signal = normaliser_signal(signal)
    
    progress_bar.update_progress(10, "Transforming audio...")
    
    coeffs = mdct(signal, frame_size=frame_size)
    QTimer.singleShot(200, lambda: continue_compression(coeffs, fs, compression_type))

def continue_compression(coeffs, fs, compression_type):
    q = quantization_factor
    qcoeffs = quantifier(coeffs, q=q)
    
    progress_bar.update_progress(30, "Compressing data...")
    
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
        qcoeffs, q_factors = quantifier_perceptual(coeffs, weights, q)
        huffman = HuffmanCoder(precision=0)
        flat_qcoeffs = qcoeffs.flatten().tolist()
        huffman_compressed = huffman.compress(flat_qcoeffs)
        lzw = LZWCoder()
        compressed_data = lzw.compress(huffman_compressed)
    elif compression_type == "Huffman + Masquage":
        psycho_model = PsychoacousticModel(fs)
        weights = psycho_model.perceptual_bit_allocation(coeffs, frame_size)
        qcoeffs, q_factors = quantifier_perceptual(coeffs, weights, q)
        huffman = HuffmanCoder(precision=0)
        flat_qcoeffs = qcoeffs.flatten().tolist()
        compressed_data = huffman.compress(flat_qcoeffs)
    elif compression_type == "LZW + Masquage":
        psycho_model = PsychoacousticModel(fs)
        weights = psycho_model.perceptual_bit_allocation(coeffs, frame_size)
        qcoeffs, q_factors = quantifier_perceptual(coeffs, weights, q)
        lzw = LZWCoder()
        compressed_data = lzw.compress(qcoeffs.flatten().tobytes())
    
    with open(compressed_file, 'wb') as f:
        header = np.array([fs, frame_size, qcoeffs.shape[0], qcoeffs.shape[1]], dtype=np.int32)
        header.tofile(f)
        f.write(compressed_data)
    
    progress_bar.update_progress(50, "Decompressing data...")
    
    with open(compressed_file, 'rb') as f:
        header = np.fromfile(f, dtype=np.int32, count=4)
        fs_loaded, frame_size_loaded, rows, cols = header
        compressed_data = f.read()
    
    QTimer.singleShot(300, lambda: decompress_data(compressed_data, compression_type, fs_loaded, frame_size_loaded, rows, cols, q))

def decompress_data(compressed_data, compression_type, fs_loaded, frame_size_loaded, rows, cols, q):
    if compression_type == "LZW only":
        lzw = LZWCoder()
        decompressed_data = lzw.decompress(compressed_data)
        qcoeffs_loaded = np.frombuffer(decompressed_data, dtype=np.int16).reshape(rows, cols)
        coeffs_recon = dequantifier(qcoeffs_loaded, q=q)
    elif compression_type == "Huffman only":
        huffman = HuffmanCoder(precision=0)
        flat_qcoeffs = huffman.decompress(compressed_data)
        qcoeffs_loaded = np.array(flat_qcoeffs, dtype=np.int16).reshape(rows, cols)
        coeffs_recon = dequantifier(qcoeffs_loaded, q=q)
    elif compression_type == "Huffman + LZW":
        lzw = LZWCoder()
        huffman_data = lzw.decompress(compressed_data)
        huffman = HuffmanCoder(precision=0)
        flat_qcoeffs = huffman.decompress(huffman_data)
        qcoeffs_loaded = np.array(flat_qcoeffs, dtype=np.int16).reshape(rows, cols)
        coeffs_recon = dequantifier(qcoeffs_loaded, q=q)
    elif compression_type in ["Huffman + LZW + Masquage", "Huffman + Masquage", "LZW + Masquage"]:
        if compression_type == "Huffman + LZW + Masquage":
            lzw = LZWCoder()
            huffman_data = lzw.decompress(compressed_data)
            huffman = HuffmanCoder(precision=0)
            flat_qcoeffs = huffman.decompress(huffman_data)
            qcoeffs_loaded = np.array(flat_qcoeffs, dtype=np.int16).reshape(rows, cols)
        elif compression_type == "Huffman + Masquage":
            huffman = HuffmanCoder(precision=0)
            flat_qcoeffs = huffman.decompress(compressed_data)
            qcoeffs_loaded = np.array(flat_qcoeffs, dtype=np.int16).reshape(rows, cols)
        else:
            lzw = LZWCoder()
            decompressed_data = lzw.decompress(compressed_data)
            qcoeffs_loaded = np.frombuffer(decompressed_data, dtype=np.int16).reshape(rows, cols)
        coeffs_recon = dequantifier(qcoeffs_loaded, q=q)
    
    progress_bar.update_progress(80, "Reconstructing audio...")
    
    signal_recon = imdct(coeffs_recon, frame_size=frame_size_loaded)
    signal_recon = normaliser_signal(signal_recon)
    
    audio_recon = signal_to_audio(signal_recon, fs_loaded)
    audio_recon.export(reconstructed_file, format="wav")
    
    QTimer.singleShot(400, lambda: finish_compression(compression_type))

def finish_compression(compression_type):
    update_stats()
    plot_comparison()
    
    compress_button.setEnabled(True)
    load_button.setEnabled(True)
    play_compressed_button.setEnabled(True)
    
    pulse_effect(stats_frame)
    file_label.setText(f"Compressed with {compression_type}\nOutput: output.irm")
    progress_bar.update_progress(100, "Compression completed!")
    
    QTimer.singleShot(2000, lambda: progress_bar.hide())

def play_compressed_audio():
    if os.path.exists(reconstructed_file):
        audio_recon = AudioSegment.from_file(reconstructed_file)
        play(audio_recon)
        highlight_widget(play_compressed_button)
    else:
        file_label.setText("No compressed file available")

def play_original_audio():
    if audio:
        play(audio)
        highlight_widget(play_original_button)
    else:
        file_label.setText("No audio file loaded")

def show_size_difference():
    if os.path.exists(original_file) and os.path.exists(compressed_file):
        original_size = os.path.getsize(original_file) / 1024
        compressed_size = os.path.getsize(compressed_file) / 1024
        diff = original_size - compressed_size
        stats_labels["difference"].setText(f"{diff:.2f} KB")
        highlight_widget(stats_frame)
    else:
        file_label.setText("Required files not available")

def show_compression_percentage():
    if os.path.exists(original_file) and os.path.exists(compressed_file):
        original_size = os.path.getsize(original_file) / 1024
        compressed_size = os.path.getsize(compressed_file) / 1024
        if original_size > 0:
            percentage = (1 - compressed_size / original_size) * 100
            stats_labels["percentage"].setText(f"{percentage:.2f}%")
            highlight_widget(stats_frame)
        else:
            file_label.setText("Invalid original size")
    else:
        file_label.setText("Required files not available")

def update_stats():
    if os.path.exists(original_file) and os.path.exists(compressed_file):
        original_size = os.path.getsize(original_file) / 1024
        compressed_size = os.path.getsize(compressed_file) / 1024
        diff = original_size - compressed_size
        percentage = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
        stats_labels["original"].setText(f"{original_size:.2f} KB")
        stats_labels["compressed"].setText(f"{compressed_size:.2f} KB")
        stats_labels["difference"].setText(f"{diff:.2f} KB")
        stats_labels["percentage"].setText(f"{percentage:.2f}%")

def plot_original_waveform():
    if audio:
        signal = normaliser_signal(audio_to_signal(audio))
        
        for i in reversed(range(waveform_preview_layout.count())):
            widget = waveform_preview_layout.itemAt(i).widget()
            if widget:
                waveform_preview_layout.removeWidget(widget)
                widget.deleteLater()
        
        fig, ax = plt.subplots(figsize=(6, 2))
        plt.style.use('dark_background')
        ax.plot(signal[:5000], color=COLORS['accent_light'], linewidth=0.8)
        ax.set_title("Audio Waveform Preview", color=COLORS['text_primary'])
        ax.set_facecolor(COLORS['panel_bg'])
        fig.patch.set_facecolor(COLORS['panel_bg'])
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 5000)
        ax.set_ylim(-1, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(COLORS['accent_dark'])
        ax.spines['left'].set_color(COLORS['accent_dark'])
        ax.tick_params(axis='x', colors=COLORS['text_secondary'])
        ax.tick_params(axis='y', colors=COLORS['text_secondary'])
        
        canvas = FigureCanvas(fig)
        waveform_preview_layout.addWidget(canvas)

def plot_comparison():
    global visualization_layout
    for i in reversed(range(visualization_layout.count())):
        widget = visualization_layout.itemAt(i).widget()
        if widget:
            visualization_layout.removeWidget(widget)
            widget.deleteLater()

    signal = normaliser_signal(audio_to_signal(audio))
    signal_recon = normaliser_signal(audio_to_signal(AudioSegment.from_file(reconstructed_file)))

    fig1, ax1 = plt.subplots(figsize=(6, 2))
    plt.style.use('dark_background')
    ax1.plot(signal[:1000], label="Original", color=COLORS['accent_light'])
    ax1.set_title("Original Signal", color=COLORS['text_primary'])
    ax1.set_facecolor(COLORS['panel_bg'])
    fig1.patch.set_facecolor(COLORS['panel_bg'])
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    fig2, ax2 = plt.subplots(figsize=(6, 2))
    plt.style.use('dark_background')
    ax2.plot(signal_recon[:1000], label="Reconstructed", color=COLORS['highlight'])
    ax2.set_title("Reconstructed Signal", color=COLORS['text_primary'])
    ax2.set_facecolor(COLORS['panel_bg'])
    fig2.patch.set_facecolor(COLORS['panel_bg'])
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    canvas1 = FigureCanvas(fig1)
    canvas2 = FigureCanvas(fig2)
    visualization_layout.addWidget(canvas1, 0, 0)
    visualization_layout.addWidget(canvas2, 0, 1)

# Main Application Setup
app = QApplication(sys.argv) if not QApplication.instance() else QApplication.instance()
app.setStyle(QStyleFactory.create("Fusion"))
app.setFont(QFont("Roboto", 11))

dark_palette = QPalette()
dark_palette.setColor(QPalette.Window, QColor(COLORS['primary_dark']))
dark_palette.setColor(QPalette.WindowText, QColor(COLORS['text_primary']))
dark_palette.setColor(QPalette.Base, QColor(COLORS['secondary_dark']))
dark_palette.setColor(QPalette.AlternateBase, QColor(COLORS['panel_bg']))
dark_palette.setColor(QPalette.ToolTipBase, QColor(COLORS['accent']))
dark_palette.setColor(QPalette.ToolTipText, QColor(COLORS['text_primary']))
dark_palette.setColor(QPalette.Text, QColor(COLORS['text_primary']))
dark_palette.setColor(QPalette.Button, QColor(COLORS['panel_bg']))
dark_palette.setColor(QPalette.ButtonText, QColor(COLORS['text_primary']))
dark_palette.setColor(QPalette.Link, QColor(COLORS['accent']))
dark_palette.setColor(QPalette.Highlight, QColor(COLORS['accent']))
dark_palette.setColor(QPalette.HighlightedText, QColor(COLORS['text_primary']))
app.setPalette(dark_palette)

QToolTip.setFont(QFont('Roboto', 11))
app.setStyleSheet(f"""
    QToolTip {{
        background-color: {COLORS['panel_bg']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['accent']};
        border-radius: 4px;
        padding: 5px;
    }}
""")

window = QWidget()
window.setWindowTitle("IRM Audio Compressor")
window.setGeometry(100, 100, 1400, 900)
window.setMinimumSize(1000, 700)

main_layout = QVBoxLayout()
main_layout.setContentsMargins(15, 15, 15, 15)
window.setLayout(main_layout)

header_frame = QFrame()
header_layout = QHBoxLayout()
header_layout.setContentsMargins(0, 0, 0, 8)
header_frame.setLayout(header_layout)

logo_label = QLabel()
logo_pixmap = QPixmap("icons/logo.png")
if not logo_pixmap.isNull():
    logo_label.setPixmap(logo_pixmap.scaled(35, 35, Qt.KeepAspectRatio, Qt.SmoothTransformation))
else:
    logo_label.setText("🎵")
    logo_label.setStyleSheet(f"font-size: 28px; color: {COLORS['accent']};")
header_layout.addWidget(logo_label)

header_label = QLabel("IRM Audio Compressor")
header_label.setStyleSheet(f"""
    font-size: 24px; 
    font-weight: bold; 
    color: {COLORS['accent']}; 
    padding-left: 8px;
""")
header_layout.addWidget(header_label)

version_label = QLabel("v2.0")
version_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px; padding-left: 8px;")
header_layout.addWidget(version_label)

header_layout.addStretch()

help_button = StyledButton("?", None)
help_button.setFixedSize(28, 28)
help_button.setStyleSheet(f"""
    QPushButton {{
        background: {COLORS['accent']};
        color: white;
        border-radius: 14px;
        font-weight: bold;
        font-size: 16px;
    }}
    QPushButton:hover {{
        background: {COLORS['accent_light']};
    }}
""")
help_button.setToolTip("Show help and documentation")
help_button.clicked.connect(lambda: HelpDialog(window).exec())
header_layout.addWidget(help_button)

main_layout.addWidget(header_frame)

tabs = QTabWidget()
tabs.setStyleSheet(f"""
    QTabWidget::pane {{
        border: 1px solid {COLORS['accent']};
        border-radius: 6px;
        margin-top: -1px;
    }}
    QTabBar::tab {{
        background: {COLORS['panel_bg']};
        color: {COLORS['text_secondary']};
        padding: 8px 18px;
        margin: 2px;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        font-size: 13px;
    }}
    QTabBar::tab:selected {{
        background: {COLORS['accent']};
        color: white;
        font-weight: bold;
    }}
    QTabBar::tab:hover:!selected {{
        background: {COLORS['secondary_dark']};
        color: {COLORS['text_primary']};
    }}
""")
main_layout.addWidget(tabs)

# Compression Tab
compression_tab = QWidget()
compression_layout = QHBoxLayout()
compression_layout.setSpacing(15)
compression_tab.setLayout(compression_layout)

left_panel = QScrollArea()
left_panel.setWidgetResizable(True)
left_panel.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
left_panel.setStyleSheet(f"""
    QScrollArea {{
        background: transparent;
        border: none;
    }}
    QScrollBar:vertical {{
        background: {COLORS['panel_bg']};
        width: 8px;
        margin: 0px;
    }}
    QScrollBar::handle:vertical {{
        background: {COLORS['accent_dark']};
        min-height: 20px;
        border-radius: 4px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
""")
left_panel.setFixedWidth(320)

left_widget = QWidget()
left_layout = QVBoxLayout()
left_layout.setSpacing(10)
left_widget.setLayout(left_layout)
left_panel.setWidget(left_widget)

# File Info Section
file_info_card = InfoCard("Audio File Information")

file_label = QLabel("No file selected")
file_label.setStyleSheet(f"""
    font-size: 13px; 
    padding: 8px; 
    background: {COLORS['secondary_dark']}; 
    border-radius: 4px; 
    color: {COLORS['text_secondary']};
""")
file_label.setAlignment(Qt.AlignCenter)
file_info_card.add_widget(file_label)

file_details_layout = QGridLayout()
file_details_layout.setColumnStretch(1, 1)
file_details_layout.setVerticalSpacing(5)

file_details_layout.addWidget(QLabel("Duration:"), 0, 0)
duration_label = QLabel("0.00 seconds")
duration_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
file_details_layout.addWidget(duration_label, 0, 1)

file_details_layout.addWidget(QLabel("Channels:"), 1, 0)
channels_label = QLabel("0")
channels_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
file_details_layout.addWidget(channels_label, 1, 1)

file_details_layout.addWidget(QLabel("Sample Rate:"), 2, 0)
sample_rate_label = QLabel("0 Hz")
sample_rate_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
file_details_layout.addWidget(sample_rate_label, 2, 1)

file_details_layout.addWidget(QLabel("Bit Depth:"), 3, 0)
bit_depth_label = QLabel("0 bits")
bit_depth_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
file_details_layout.addWidget(bit_depth_label, 3, 1)

file_info_card.add_layout(file_details_layout)
left_layout.addWidget(file_info_card)

# Controls Section
controls_card = InfoCard("Compression Controls")

drop_area = DropArea()
controls_card.add_widget(drop_area)

load_button = StyledButton("Select File", "icons/load.png")
load_button.clicked.connect(lambda: load_audio_file(QFileDialog.getOpenFileName(None, "Open Audio File", "", "Audio Files (*.wav *.mp3 *.flac *.ogg)")[0]))
controls_card.add_widget(load_button)

recent_files_label = QLabel("Recent Files:")
recent_files_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
controls_card.add_widget(recent_files_label)

recent_files_combo = QComboBox()
recent_files_combo.setStyleSheet(f"""
    QComboBox {{
        background: {COLORS['secondary_dark']};
        color: {COLORS['text_primary']};
        padding: 8px;
        border-radius: 4px;
        font-size: 12px;
    }}
    QComboBox::drop-down {{
        border: none;
        width: 25px;
    }}
    QComboBox QAbstractItemView {{
        background: {COLORS['secondary_dark']};
        color: {COLORS['text_primary']};
        selection-background-color: {COLORS['accent']};
    }}
""")
recent_files_combo.setToolTip("Select a recently used file")
controls_card.add_widget(recent_files_combo)

compression_label = QLabel("Compression Method:")
compression_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px; margin-top: 10px;")
controls_card.add_widget(compression_label)

compression_combo = QComboBox()
compression_combo.addItems([
    "LZW only",
    "Huffman only",
    "Huffman + LZW",
    "Huffman + LZW + Masquage",
    "Huffman + Masquage",
    "LZW + Masquage"
])
compression_combo.setStyleSheet(f"""
    QComboBox {{
        background: {COLORS['secondary_dark']};
        color: {COLORS['text_primary']};
        padding: 8px;
        border-radius: 4px;
        font-size: 12px;
    }}
    QComboBox::drop-down {{
        border: none;
        width: 25px;
    }}
    QComboBox QAbstractItemView {{
        background: {COLORS['secondary_dark']};
        color: {COLORS['text_primary']};
        selection-background-color: {COLORS['accent']};
    }}
""")
compression_combo.setToolTip("Choose a compression method")
controls_card.add_widget(compression_combo)

compress_button = StyledButton("Compress", "icons/compress.png")
compress_button.clicked.connect(compress_audio)
compress_button.setEnabled(False)
controls_card.add_widget(compress_button)

playback_layout = QHBoxLayout()
playback_layout.setSpacing(10)

play_original_button = StyledButton("Play Original", "icons/play.png")
play_original_button.clicked.connect(play_original_audio)
play_original_button.setEnabled(False)
playback_layout.addWidget(play_original_button)

play_compressed_button = StyledButton("Play Compressed", "icons/play.png")
play_compressed_button.clicked.connect(play_compressed_audio)
play_compressed_button.setEnabled(False)
playback_layout.addWidget(play_compressed_button)

controls_card.add_layout(playback_layout)

progress_bar = EnhancedProgressBar()
progress_bar.hide()
controls_card.add_widget(progress_bar)

left_layout.addWidget(controls_card)

# Stats Section
stats_card = InfoCard("Compression Statistics")
stats_frame = QFrame()
stats_layout = QGridLayout()
stats_layout.setVerticalSpacing(5)
stats_frame.setLayout(stats_layout)

stats_labels = {}
labels = ["Original Size:", "Compressed Size:", "Difference:", "Compression Rate:"]
keys = ["original", "compressed", "difference", "percentage"]
for i, (label, key) in enumerate(zip(labels, keys)):
    lbl = QLabel(label)
    lbl.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
    stats_layout.addWidget(lbl, i, 0)
    stats_labels[key] = QLabel("0.00")
    stats_labels[key].setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 12px;")
    stats_layout.addWidget(stats_labels[key], i, 1)

stats_card.add_widget(stats_frame)

stats_buttons_layout = QHBoxLayout()
stats_buttons_layout.setSpacing(10)

size_diff_button = StyledButton("Show Difference", "icons/size.png")
size_diff_button.clicked.connect(show_size_difference)
stats_buttons_layout.addWidget(size_diff_button)

percentage_button = StyledButton("Show %", "icons/percentage.png")
percentage_button.clicked.connect(show_compression_percentage)
stats_buttons_layout.addWidget(percentage_button)

stats_card.add_layout(stats_buttons_layout)
left_layout.addWidget(stats_card)

left_layout.addStretch()
compression_layout.addWidget(left_panel)

# Right Panel (Visualization)
right_panel = QFrame()
right_layout = QVBoxLayout()
right_layout.setSpacing(10)
right_panel.setLayout(right_layout)

visualization_label = QLabel("Visualization")
visualization_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {COLORS['accent']}; margin: 5px;")
visualization_label.setAlignment(Qt.AlignCenter)
right_layout.addWidget(visualization_label)

waveform_card = InfoCard("Waveform Preview")
waveform_preview_layout = QVBoxLayout()
waveform_card.add_layout(waveform_preview_layout)
right_layout.addWidget(waveform_card)

visualization_card = InfoCard("Compression Analysis")
visualization_layout = QGridLayout()
visualization_layout.setSpacing(10)
visualization_card.add_layout(visualization_layout)
right_layout.addWidget(visualization_card)

compression_layout.addWidget(right_panel)
tabs.addTab(compression_tab, "Compression")

window.show()
sys.exit(app.exec())
