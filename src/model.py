import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import joblib
import os

class AdvancedSpeakerRecognitionModel:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.scaler = StandardScaler()
        self.svm = None
        self.nn = None
        self.rf = None
        self.ensemble = None
        self.speakers = []

    def preprocess_features(self, features):
        return self.scaler.transform(features)

    def train(self, features, labels):
        # Ensure features is 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        elif features.ndim > 2:
            features = features.reshape(features.shape[0], -1)

        self.speakers = list(set(labels))
        self.scaler.fit(features)
        X_scaled = self.preprocess_features(features)

        X_train, X_val, y_train, y_val = train_test_split(X_scaled, labels, test_size=0.2, random_state=42)

        self.svm = SVC(kernel='rbf', probability=True, random_state=42)
        self.nn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42)

        self.ensemble = VotingClassifier(
            estimators=[
                ('svm', self.svm),
                ('nn', self.nn),
                ('rf', self.rf)
            ],
            voting='soft'
        )

        self.ensemble.fit(X_train, y_train)

        accuracy = self.ensemble.score(X_val, y_val)
        print(f"Ensemble Model Accuracy: {accuracy:.2f}")

    def predict_speaker(self, features):
        if features.ndim == 1:
            features = features.reshape(1, -1)
        elif features.ndim > 2:
            features = features.reshape(features.shape[0], -1)

        X_scaled = self.preprocess_features(features)
        probabilities = self.ensemble.predict_proba(X_scaled)
        speaker_index = np.argmax(np.mean(probabilities, axis=0))
        confidence = np.max(np.mean(probabilities, axis=0))
        
        predicted_speaker = self.speakers[speaker_index]
        return predicted_speaker, confidence

    def save_model(self, filename):
        model_path = os.path.join(self.model_dir, filename)
        joblib.dump({
            'scaler': self.scaler,
            'svm': self.svm,
            'nn': self.nn,
            'rf': self.rf,
            'ensemble': self.ensemble,
            'speakers': self.speakers
        }, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, filename):
        model_path = os.path.join(self.model_dir, filename)
        loaded_model = joblib.load(model_path)
        self.scaler = loaded_model['scaler']
        self.svm = loaded_model['svm']
        self.nn = loaded_model['nn']
        self.rf = loaded_model['rf']
        self.ensemble = loaded_model['ensemble']
        self.speakers = loaded_model['speakers']
        print(f"Model loaded from {model_path}")

if __name__ == "__main__":
    # Test the model
    model = AdvancedSpeakerRecognitionModel()
    
    # Generate dummy data
    np.random.seed(42)
    n_samples = 100
    n_features = 32  # Adjust this to match your actual feature dimension
    n_speakers = 3
    
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, n_speakers, n_samples)
    
    # Train the model
    model.train(X, y)
    
    # Test prediction
    test_sample = np.random.rand(1, n_features)
    predicted_speaker, confidence = model.predict_speaker(test_sample)
    print(f"Predicted speaker: {predicted_speaker}, Confidence: {confidence:.2f}")