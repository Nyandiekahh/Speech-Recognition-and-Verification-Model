import librosa
import numpy as np

def extract_mfcc(audio, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfccs.T  # Transpose to get time as the first dimension

def extract_spectral_features(audio, sr):
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)[0]
    
    # Ensure all spectral features have the same length
    target_length = len(spectral_centroids)
    spectral_rolloff = librosa.util.fix_length(spectral_rolloff, size=target_length)
    spectral_contrast = librosa.util.fix_length(spectral_contrast, size=target_length)
    
    spectral_features = np.vstack([spectral_centroids, spectral_rolloff, spectral_contrast])
    return spectral_features.T  # Transpose to get time as the first dimension

def extract_features(audio_path):
    audio, sr = librosa.load(audio_path)
    mfccs = extract_mfcc(audio, sr)
    spectral = extract_spectral_features(audio, sr)
    
    # Ensure MFCCs and spectral features have the same number of time steps
    min_length = min(mfccs.shape[0], spectral.shape[0])
    mfccs = mfccs[:min_length, :]
    spectral = spectral[:min_length, :]
    
    # Combine features
    combined_features = np.hstack((mfccs, spectral))
    
    # Calculate mean and standard deviation for each feature
    feature_means = np.mean(combined_features, axis=0)
    feature_stds = np.std(combined_features, axis=0)
    
    # Return a 1D array of means and standard deviations
    return np.hstack((feature_means, feature_stds))

if __name__ == "__main__":
    # Test the feature extraction
    audio_file = "path/to/your/test/audio/file.wav"
    features = extract_features(audio_file)
    print(f"Extracted features shape: {features.shape}")