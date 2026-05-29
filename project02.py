import os
import numpy as np
import pandas as pd
import librosa

def load_music_data(folder_path, n_mfcc=13):
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

                # Extract genre from filename
                # Example: blues00000.wav -> blues
                genre = ''.join([char for char in file_name if char.isalpha()])

                # Store data
                features.append(mfccs_mean)
                labels.append(genre)

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)

    return X, y