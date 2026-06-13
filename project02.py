#%%
#Data import function

import os
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
import matplotlib.pyplot as plt

def load_music_data(folder_path, n_mfcc=20):
    """
    Loads .wav files, extracts MFCC features,
    and stores the genre label from the filename.

    Example filename:
    blues00000.wav -> genre = blues
    """

    features = []
    labels = []

    # Loop through all files in the folder
    for file_name in os.listdir(folder_path):

        # Only process .wav files
        if file_name.endswith(".wav"):

            file_path = os.path.join(folder_path, file_name)

            try:
                # Load audio file
                audio, sample_rate = librosa.load(file_path, sr=None)

                # Extract MFCC features
                mfccs = librosa.feature.mfcc(
                    y=audio,
                    sr=sample_rate,
                    n_mfcc=n_mfcc
                )

                # Take the mean of each MFCC coefficient
                mfccs_mean = np.mean(mfccs.T, axis=0)
                #Take the variance of each MFCC coefficient
                mfccs_var = np.var(mfccs, axis=1)
                #Combine features (horizontally for classifier)
                feature_vector = np.hstack((mfccs_mean, mfccs_var))
                # Extract genre from filename
                # Example: blues00000.wav -> blues
                genre = ''.join([char for char in file_name if char.isalpha()])

                # Store data
                features.append(feature_vector)
                labels.append(genre)

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)

    return X, y

#%%
folder_path = Path.cwd() / "Data" / "genres_original"
n_mfcc=13 #Feature number ~ hyperparameter
X, y = load_music_data(folder_path, n_mfcc) 
print(len(y))
print(np.shape(X))

#%%
plt.imshow((X[:, :n_mfcc].T), cmap='viridis', origin='lower', aspect=40)
plt.title("Averaged MFCCs")
plt.xlabel("Song number")
plt.ylabel("MFCC coefficient")
plt.show()
plt.imshow((X[:, n_mfcc:].T), cmap='viridis', origin='lower', aspect=40)
plt.title("Variance of MFCCs")
plt.xlabel("Song number")
plt.ylabel("MFCC coefficient")
plt.show()

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=67, 
    stratify=y
)

scaler = StandardScaler()

# "Fit" learns the mean/variance of the training data, "transform" applies it
X_train_scaled = scaler.fit_transform(X_train)

# ONLY "transform" the test data. Never "fit" on test data (that is cheating!)
X_test_scaled = scaler.transform(X_test)

# --- Step 3: Initialize and Train the Model ---
# Set k=5 as a standard starting point
knn = KNeighborsClassifier(n_neighbors=25, metric='euclidean') #Hyperparameter

# Train the model on the scaled data
knn.fit(X_train_scaled, y_train)

# --- Step 4: Make Predictions and Evaluate ---
y_pred = knn.predict(X_test_scaled)

# Print out the results
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")

# The classification report shows exactly which genres the model is struggling with
print("Detailed Classification Report:")
print(classification_report(y_test, y_pred))
# %%
