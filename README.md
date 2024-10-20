# Advanced Speec Recognition and verification System

## Overview

This project implements an advanced speaker recognition system using machine learning techniques. The system can record audio samples from multiple users, extract relevant features, train a model to recognize these users, and then verify the identity of a speaker based on new audio input.

## Features

- User Management: Add and manage multiple users
- Audio Recording: Record audio samples directly through the application
- Audio Upload: Support for uploading existing audio files (.wav, .mp3, .ogg, .flac)
- Feature Extraction: Extract MFCC and spectral features from audio samples
- Model Training: Train an ensemble model using SVM, Neural Network, and Random Forest classifiers
- Speaker Verification: Verify the identity of a speaker using recorded or uploaded audio
- User-friendly GUI: Easy-to-use graphical interface for all operations

## Technologies Used

- Python 3.x
- tkinter for GUI
- librosa for audio processing and feature extraction
- scikit-learn for machine learning models
- pydub for audio file conversion
- numpy for numerical operations
- joblib for model serialization

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/nyandiekahh/speaker-recognition-project.git
   cd speaker-recognition-project
   ```

2. Create and activate a virtual environment:
   ```
   python3 -m venv myenv
   source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install numpy scipy librosa scikit-learn joblib pydub
   ```

4. Install FFmpeg (required for audio conversion):
   - On Ubuntu or Debian:
     ```
     sudo apt-get update
     sudo apt-get install ffmpeg
     ```
   - On macOS (using Homebrew):
     ```
     brew install ffmpeg
     ```
   - On Windows, download from the official FFmpeg website and add it to your system PATH.

## Project Structure

```
speaker-recognition-project/
│
├── src/
│   ├── gui.py
│   ├── model.py
│   ├── feature_extraction.py
│   └── data_collection.py
│
├── data/
│   └── speech_samples/
│
├── models/
│
├── requirements.txt
└── README.md
```

## Usage

1. Run the application:
   ```
   python3 src/gui.py
   ```

2. Using the GUI:
   - Add Users: Enter a username and click "Add User"
   - Record Audio: Select a user and click "Record" to record 5 audio samples
   - Upload Audio: Select a user and click "Upload" to add existing audio files
   - Train Model: After adding audio for at least two users, click "Train Model"
   - Verify Speaker: Use "Record and Verify" or "Upload and Verify" to test the model

## Model Details

The speaker recognition model uses an ensemble of three classifiers:
- Support Vector Machine (SVM)
- Neural Network (Multi-layer Perceptron)
- Random Forest

These models are combined using a soft voting classifier to make the final prediction.

## Feature Extraction

Audio features are extracted using the librosa library. The system uses:
- Mel-frequency cepstral coefficients (MFCCs)
- Spectral centroid
- Spectral rolloff
- Spectral contrast

## Limitations and Future Work

- The current system is designed for a limited number of known speakers. Scaling to a larger number of speakers may require additional optimization.
- The system assumes relatively clean audio. Performance may degrade with noisy recordings or poor quality microphones.
- Future work could include:
  - Implementing speaker diarization for multi-speaker audio
  - Adding support for real-time speaker identification
  - Exploring deep learning models like Convolutional Neural Networks (CNNs) or Recurrent Neural Networks (RNNs) for feature extraction and classification

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

[MIT License](https://opensource.org/licenses/MIT)

## Contact

For any questions or feedback, please contact [Your Name] at [einsteinmokua100@gmail.com].