import os
import numpy as np
from data_collection import record_audio
from feature_extraction import extract_features
from model import AdvancedSpeakerRecognitionModel
import warnings
import contextlib

# Suppress ALSA warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sounddevice")

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = os.sys.stdout
        os.sys.stdout = devnull
        try:  
            yield
        finally:
            os.sys.stdout = old_stdout

def main():
    data_dir = "data/speech_samples"
    model_dir = "models"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Initialize the model
    model = AdvancedSpeakerRecognitionModel(model_dir)

    # Collect data
    users = ["user1", "user2", "user3"]  # Add more users as needed
    features = []
    labels = []

    for user in users:
        user_dir = os.path.join(data_dir, user)
        os.makedirs(user_dir, exist_ok=True)
        print(f"Recording audio for {user}")
        try:
            with suppress_stdout():
                record_audio(user_dir, num_samples=5)
        except Exception as e:
            print(f"Error recording audio for {user}: {str(e)}")
            continue

        # Extract features
        for audio_file in os.listdir(user_dir):
            if audio_file.endswith(".wav"):
                audio_path = os.path.join(user_dir, audio_file)
                try:
                    feature = extract_features(audio_path)
                    features.append(feature)
                    labels.append(user)
                except Exception as e:
                    print(f"Error extracting features from {audio_file}: {str(e)}")
                    continue

    if not features:
        print("No features extracted. Exiting.")
        return

    features = np.array(features)
    labels = np.array(labels)

    # Train the model
    try:
        model.train(features, labels)
        model.save_model("speaker_recognition_model.joblib")
        print("Model trained and saved successfully.")
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return

    # Verify a speaker
    test_user = "user1"  # Change this to test different users
    test_audio = os.path.join(data_dir, test_user, "sample_1.wav")  # Assuming this file exists
    if not os.path.exists(test_audio):
        print(f"Test audio file not found: {test_audio}")
        return

    try:
        test_features = extract_features(test_audio)
    except Exception as e:
        print(f"Error extracting features from test audio: {str(e)}")
        return

    try:
        predicted_speaker, confidence = model.predict_speaker(test_features)
        print(f"Actual speaker: {test_user}")
        print(f"Predicted speaker: {predicted_speaker}")
        print(f"Confidence: {confidence:.2f}")
    except Exception as e:
        print(f"Error verifying speaker: {str(e)}")

if __name__ == "__main__":
    main()