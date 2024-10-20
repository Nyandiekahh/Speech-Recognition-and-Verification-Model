import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import numpy as np
import shutil
from pydub import AudioSegment
from data_collection import record_audio
from feature_extraction import extract_features
from model import AdvancedSpeakerRecognitionModel

class SpeechRecognitionApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Advanced Speech Recognition and Verification")
        self.geometry("500x400")

        self.users = []
        self.current_user = tk.StringVar()
        self.model_dir = "models"
        self.data_dir = "data/speech_samples"
        self.model = AdvancedSpeakerRecognitionModel(self.model_dir)

        self.create_widgets()

    def create_widgets(self):
        # User management
        user_frame = ttk.LabelFrame(self, text="User Management")
        user_frame.pack(padx=10, pady=10, fill="x")

        ttk.Label(user_frame, text="Username:").pack(side="left", padx=5)
        self.user_entry = ttk.Entry(user_frame)
        self.user_entry.pack(side="left", padx=5)
        ttk.Button(user_frame, text="Add User", command=self.add_user).pack(side="left", padx=5)

        # Recording and Upload
        record_frame = ttk.LabelFrame(self, text="Record/Upload Audio")
        record_frame.pack(padx=10, pady=10, fill="x")

        ttk.Label(record_frame, text="Select User:").pack(side="left", padx=5)
        self.user_combo = ttk.Combobox(record_frame, textvariable=self.current_user)
        self.user_combo.pack(side="left", padx=5)
        self.user_combo['values'] = self.users

        ttk.Button(record_frame, text="Record", command=self.record_audio).pack(side="left", padx=5)
        ttk.Button(record_frame, text="Upload", command=self.upload_audio).pack(side="left", padx=5)

        # Training
        train_frame = ttk.LabelFrame(self, text="Train Model")
        train_frame.pack(padx=10, pady=10, fill="x")

        ttk.Button(train_frame, text="Train Model", command=self.train_model).pack(padx=5, pady=5)

        # Verification
        verify_frame = ttk.LabelFrame(self, text="Verify Speaker")
        verify_frame.pack(padx=10, pady=10, fill="x")

        ttk.Button(verify_frame, text="Record and Verify", command=self.verify_speaker).pack(side="left", padx=5, pady=5)
        ttk.Button(verify_frame, text="Upload and Verify", command=self.upload_and_verify).pack(side="left", padx=5, pady=5)

    def add_user(self):
        username = self.user_entry.get()
        if username and username not in self.users:
            self.users.append(username)
            self.user_combo['values'] = self.users
            self.user_entry.delete(0, 'end')
            messagebox.showinfo("Success", f"User {username} added successfully!")
        else:
            messagebox.showerror("Error", "Invalid username or user already exists!")

    def record_audio(self):
        user = self.current_user.get()
        if user:
            user_dir = os.path.join(self.data_dir, user)
            os.makedirs(user_dir, exist_ok=True)
            record_audio(user_dir, num_samples=5)
            messagebox.showinfo("Success", f"Recorded 5 audio samples for {user}")
        else:
            messagebox.showerror("Error", "Please select a user!")

    def convert_to_wav(self, input_file, output_file):
        audio = AudioSegment.from_file(input_file)
        audio.export(output_file, format="wav")

    def upload_audio(self):
        user = self.current_user.get()
        if user:
            user_dir = os.path.join(self.data_dir, user)
            os.makedirs(user_dir, exist_ok=True)
            filetypes = [('Audio files', '*.wav *.mp3 *.ogg *.flac'), ('All files', '*.*')]
            files = filedialog.askopenfilenames(title="Select audio files", filetypes=filetypes)
            if files:
                converted_count = 0
                for file in files:
                    file_name, file_extension = os.path.splitext(file)
                    if file_extension.lower() != '.wav':
                        wav_file = os.path.join(user_dir, f"{os.path.basename(file_name)}.wav")
                        self.convert_to_wav(file, wav_file)
                        converted_count += 1
                    else:
                        shutil.copy(file, user_dir)
                
                total_files = len(files)
                messagebox.showinfo("Success", f"Uploaded {total_files} audio samples for {user}.\n{converted_count} files were converted to WAV format.")
        else:
            messagebox.showerror("Error", "Please select a user!")

    def train_model(self):
        features = []
        labels = []
        for user in self.users:
            user_dir = os.path.join(self.data_dir, user)
            if not os.path.exists(user_dir):
                continue
            for audio_file in os.listdir(user_dir):
                if audio_file.endswith(".wav"):
                    audio_path = os.path.join(user_dir, audio_file)
                    try:
                        feature = extract_features(audio_path)
                        print(f"Feature shape for {audio_file}: {feature.shape}")  # Debug print
                        features.append(feature)
                        labels.append(user)
                    except Exception as e:
                        print(f"Error extracting features from {audio_file}: {str(e)}")

        if not features:
            messagebox.showwarning("Warning", "No audio samples found. Please record or upload audio for users.")
            return

        features = np.array(features)
        labels = np.array(labels)

        print(f"Final features shape: {features.shape}")  # Debug print
        print(f"Labels shape: {labels.shape}")  # Debug print

        try:
            self.model.train(features, labels)
            self.model.save_model("speaker_recognition_model.joblib")
            messagebox.showinfo("Success", "Model trained and saved successfully!")
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            messagebox.showerror("Error", f"Model training failed: {str(e)}")

    def verify_speaker(self):
        verify_dir = os.path.join(self.data_dir, "verification")
        os.makedirs(verify_dir, exist_ok=True)
        record_audio(verify_dir, num_samples=1)
        
        test_audio = os.path.join(verify_dir, "sample_1.wav")
        self.perform_verification(test_audio)

    def upload_and_verify(self):
        filetypes = [('Audio files', '*.wav *.mp3 *.ogg *.flac'), ('All files', '*.*')]
        file = filedialog.askopenfilename(title="Select audio file for verification", filetypes=filetypes)
        if file:
            file_name, file_extension = os.path.splitext(file)
            if file_extension.lower() != '.wav':
                temp_wav = os.path.join(self.data_dir, "temp_verify.wav")
                self.convert_to_wav(file, temp_wav)
                self.perform_verification(temp_wav)
                os.remove(temp_wav)  # Clean up the temporary file
            else:
                self.perform_verification(file)
        else:
            messagebox.showwarning("Warning", "No file selected for verification")

    def perform_verification(self, audio_file):
        try:
            test_features = extract_features(audio_file)
            predicted_speaker, confidence = self.model.predict_speaker(test_features)
            
            result_message = f"Predicted Speaker: {predicted_speaker}\nConfidence: {confidence:.2f}"
            messagebox.showinfo("Verification Result", result_message)
        except Exception as e:
            print(f"Error during verification: {str(e)}")
            messagebox.showerror("Error", f"Verification failed: {str(e)}")

if __name__ == "__main__":
    app = SpeechRecognitionApp()
    app.mainloop()