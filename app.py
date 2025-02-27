import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import io
from pydub import AudioSegment

# Streamlit UI
st.title("🔊 AI-Powered DJ - Mood-Based Auto Mixer")

# File Upload
uploaded_file = st.file_uploader("Upload a music file (MP3, WAV)", type=["mp3", "wav"])

if uploaded_file:
    # Convert file to WAV (if needed)
    file_type = uploaded_file.name.split(".")[-1]
    if file_type == "mp3":
        audio = AudioSegment.from_mp3(uploaded_file)
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        buffer.seek(0)
        y, sr = librosa.load(buffer, sr=None)
    else:
        y, sr = librosa.load(uploaded_file, sr=None)

    # Analyze BPM
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env)

    # Display BPM
    st.subheader(f"🎵 Estimated BPM: {int(tempo)} BPM")

    # Visualize Waveform
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.7)
    ax.set_title("Waveform")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # Mood Detection (Placeholder)
    mood = "Happy 😃" if tempo > 120 else "Chill 😌"
    st.subheader(f"🎭 Detected Mood: {mood}")

    # Future Features: Auto-Mixing & Beat Matching
    st.write("🚀 Coming soon: AI-powered mixing, crossfading & genre-based recommendations!")
