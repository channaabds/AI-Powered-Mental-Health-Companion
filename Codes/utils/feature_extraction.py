import librosa
import numpy as np

def get_audio_features(audio_data, sampling_rate):
    # Ensure audio data is floating point and mono
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Perform harmonic-percussive source separation
    y_harmonic, y_percussive = librosa.effects.hpss(audio_data)
    
    # Extract pitch features
    pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sampling_rate)
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=13)
    mfccs = np.mean(mfccs, axis=1)
    
    # Ensure MFCC length is 65
    if len(mfccs) > 65:
        mfccs = mfccs[:65]
    elif len(mfccs) < 65:
        mfccs = np.pad(mfccs, (0, 65 - len(mfccs)))
    
    # Process pitch and magnitude features
    pitches = np.mean(pitches, axis=1)[:20]
    magnitudes = np.mean(magnitudes, axis=1)[:20]
    
    # Extract chroma features from harmonic component
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sampling_rate)
    chroma = np.mean(chroma, axis=1)
    
    return mfccs, pitches, magnitudes, chroma