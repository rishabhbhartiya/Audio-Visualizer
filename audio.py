import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
st.header("VISUALIZE YOUR AUDIO SAMPLE")
# Display an uploader widget for audio files
uploaded_file = st.file_uploader("Upload your audio file", type=['mp3', 'wav', 'ogg'])

if uploaded_file is not None:
    # Play the uploaded audio file
    st.audio(uploaded_file, format='audio/mpeg')

    # Optionally, you can also save the uploaded file to disk
    with open("uploaded_audio.mp3", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.title("SIGNAL REPRESENTATION")
    st.write("")
    y, sr = librosa.load(uploaded_file, sr=None)
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Audio Waveform")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    
    
    
    st.title("MFCC REPRESENTATION")
    st.write("")
    st.subheader("CHROMAGRAM")
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    hop_length = 512
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='twilight')
    plt.colorbar()
    plt.title('MFCC')
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficient')
    st.pyplot()
    
    
    
    st.write("")
    st.subheader("DECIBELS SCALED SPECTROGRAM")
    n_fft = 2048 # FFT window size
    hop_length = 512
    spectrogram = np.abs(librosa.stft(y= y, n_fft = n_fft, hop_length = hop_length))
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Decibel-scaled Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    st.pyplot()
    

    st.write("")
    st.subheader("MEL SPECTROGRAM")
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    amplitude_to_db = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(amplitude_to_db,sr=sr, hop_length = hop_length, x_axis = 'time', 
                         y_axis = 'log', cmap = 'rainbow')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency (Mel)')
    st.pyplot()
    
    st.write("")
    st.subheader("HARMONICS AND PERCEPTUAL")
    audio, _ = librosa.effects.trim(y)
    y_harm, y_perc = librosa.effects.hpss(audio)
    plt.figure(figsize = (4, 3))
    plt.plot(y_perc, color = '#FFB100')
    plt.plot(y_harm, color = '#A300F9')
    plt.title("Harmonics and Perceptrual", fontsize=8)
    plt.legend(("Perceptrual", "Harmonics"))
    st.pyplot()
    
    
    
    st.write("")
    st.subheader("SPECTRAL CENTROID")
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    plt.figure(figsize=(10, 4))
    plt.plot(spectral_centroid, label='Spectral Centroid', color='b')
    plt.title('Spectral Centroid')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.legend()
    st.pyplot()
    
    st.write("")
    st.subheader("CHROMAGRAM STFT")
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma_stft, sr=sr, x_axis='time', y_axis='chroma', cmap='coolwarm')
    plt.colorbar()
    plt.title('Chroma STFT')
    plt.xlabel('Time')
    plt.ylabel('Pitch Class')
    st.pyplot()
    
    
    st.write("")
    st.subheader("SPECTRAL ROLLOFF")
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    plt.figure(figsize=(10, 4))
    plt.plot(spectral_rolloff, label='Spectral Rolloff', color='b')
    plt.title('Spectral Rolloff')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.legend()
    st.pyplot()
