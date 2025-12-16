# Emotion Recognition from Speech using Deep Learning
# GitHub-ready Machine Learning / Deep Learning Project
# Author: Student Project
# Dataset: RAVDESS / TESS / EMO-DB

# ================================
# 1. Import Required Libraries
# ================================
import os
import numpy as np
import librosa
import librosa.display

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import to_categorical

# ================================
# 2. Feature Extraction (MFCC)
# ================================
def extract_mfcc(file_path, n_mfcc=40):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled

# ================================
# 3. Load Dataset
# ================================
def load_data(dataset_path):
    features = []
    labels = []

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.wav'):
                emotion = file.split('-')[2]  # RAVDESS emotion label position
                file_path = os.path.join(root, file)
                mfcc = extract_mfcc(file_path)
                features.append(mfcc)
                labels.append(emotion)

    return np.array(features), np.array(labels)

# ================================
# 4. Dataset Path
# ================================
# Example: dataset/RAVDESS/
DATASET_PATH = 'dataset'
X, y = load_data(DATASET_PATH)

print("Feature Shape:", X.shape)

# ================================
# 5. Label Encoding
# ================================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# ================================
# 6. Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# Reshape for LSTM (samples, timesteps, features)
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# ================================
# 7. Build LSTM Model
# ================================
model = Sequential()
model.add(LSTM(128, return_sequences=False, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y_train.shape[1], activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# ================================
# 8. Train Model
# ================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32
)

# ================================
# 9. Evaluate Model
# ================================
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)

# ================================
# 10. Predict Emotion from New Audio
# ================================
def predict_emotion(file_path):
    mfcc = extract_mfcc(file_path)
    mfcc = np.expand_dims(mfcc, axis=0)
    mfcc = np.expand_dims(mfcc, axis=2)

    prediction = model.predict(mfcc)
    emotion = label_encoder.inverse_transform([np.argmax(prediction)])
    return emotion[0]

# Example prediction
# print(predict_emotion('test_audio.wav'))

# ================================
# End of Project Code
# ================================
