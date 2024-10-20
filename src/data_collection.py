import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os

def record_audio(output_dir, duration=5, sample_rate=44100, channels=1, num_samples=5):
    for i in range(num_samples):
        filename = os.path.join(output_dir, f"sample_{i+1}.wav")
        
        print(f"Recording sample {i+1}...")
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels)
        sd.wait()
        print("Finished recording.")
        
        # Normalize the audio data
        recording = np.int16(recording / np.max(np.abs(recording)) * 32767)
        
        wav.write(filename, sample_rate, recording)

if __name__ == "__main__":
    output_dir = "data/speech_samples"
    os.makedirs(output_dir, exist_ok=True)
    record_audio(output_dir)