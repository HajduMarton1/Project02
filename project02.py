# %%
# Data import function

import os
import numpy as np
import pandas as pd
import librosa
import warnings

def load_music_data(folder_path, n_mfcc=13):
    """
    Loads .wav files, extracts MFCC features,
    and stores the genre label from the filename.

    Example filename:
    blues00000.wav -> genre = blues
    """

    features = []
    labels = []
    song_ids = []  # Track the source file

    # Ignore warnings from Librosa about silent audio segments
    warnings.filterwarnings('ignore', category=UserWarning)

    # Loop through all files in the folder
    for file_name in os.listdir(folder_path):

        # Only process .wav files
        if file_name.endswith(".wav"):

            file_path = os.path.join(folder_path, file_name)
            segment_duration = 3  # in seconds

            try:
                # Load audio file
                audio, sample_rate = librosa.load(file_path, sr=None)

                #Calculate samples per segment
                samples_per_segment = int(segment_duration * sample_rate)

                #Loop through 3-second parts
                for i in range(0, len(audio), samples_per_segment):
                    
                    #Split segment
                    segment = audio[i : i + samples_per_segment]
                    
                    #Only process correctly sized segments
                    if len(segment) == samples_per_segment:
                        
                        # 1. Base MFCCs (Timbre)
                        mfccs = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=n_mfcc)
                        mfccs_mean = np.mean(mfccs.T, axis=0)
                        mfccs_var = np.var(mfccs.T, axis=0)
                        
                        """# 2. Delta & Delta-Delta MFCCs (Temporal change of Timbre)
                        delta_mfccs = librosa.feature.delta(mfccs)
                        delta_mfccs_mean = np.mean(delta_mfccs.T, axis=0)
                        delta_mfccs_var = np.var(delta_mfccs.T, axis=0)
                        
                        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
                        delta2_mfccs_mean = np.mean(delta2_mfccs.T, axis=0)
                        delta2_mfccs_var = np.var(delta2_mfccs.T, axis=0)"""
                        
                        # 3. Chroma STFT (Harmony / Pitch Classes)
                        chroma = librosa.feature.chroma_stft(y=segment, sr=sample_rate)
                        chroma_mean = np.mean(chroma.T, axis=0)
                        chroma_var = np.var(chroma.T, axis=0)
                        
                        # 4. Spectral Centroid (Brightness of Sound)
                        centroid = librosa.feature.spectral_centroid(y=segment, sr=sample_rate)
                        centroid_mean = np.mean(centroid.T, axis=0)
                        centroid_var = np.var(centroid.T, axis=0)
                        
                        """# 5. Tempo / BPM (Rhythm)
                        tempo_data, _ = librosa.beat.beat_track(y=segment, sr=sample_rate)
                        # Wrap in np.array to ensure it stacks cleanly horizontally
                        tempo = np.array([np.mean(tempo_data)])"""

                        # Combine all features into one row
                        feature_vector = np.hstack((
                            mfccs_mean, mfccs_var,
                            #delta_mfccs_mean, delta_mfccs_var,
                            #delta2_mfccs_mean, delta2_mfccs_var,
                            chroma_mean, chroma_var,
                            centroid_mean, centroid_var,
                            #tempo
                        ))
                        
                        genre = ''.join([char for char in file_name if char.isalpha()]).replace("wav", "")

                        features.append(feature_vector)
                        labels.append(genre)
                        song_ids.append(file_name)

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    groups = np.array(song_ids)

    return X, y, groups

# %%
# Data reading & proccessing

from pathlib import Path

folder_path = Path.cwd() / "Data" / "genres_original"
n_mfcc=15 #Mfcc feature number ~ hyperparameter
X, y, groups = load_music_data(folder_path, n_mfcc) 
print(len(y))
print(np.shape(X))

# %%
# Data splitting

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GroupShuffleSplit # For protection against
# target leakage

# Split the data, keeping segments of the same song together
gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=50)

# gss.split yields indices for train and test
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
groups_train, groups_test = groups[train_idx], groups[test_idx]

# %%
# kNN model

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold #Against target leakage

scaler = StandardScaler()

# "Fit" learns the mean/variance of the training data, "transform" applies it
X_train_scaled = scaler.fit_transform(X_train)

# This searches for the best parameters number using cv-fold cross validation - Hyperparameter?
param_grid = {
    'n_neighbors': 3 + 2*np.arange(15),
    'metric': ['euclidean', 'minkowski', 'cosine', 'hamming'],
    'weights': ['uniform', 'distance']
    }

# GroupKFold split with 5 splits
gkf = GroupKFold(n_splits=5)

grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=gkf, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train, groups=groups_train)

print(f"Best K Value: {grid_search.best_params_['n_neighbors']}")
print(f"Best distance type: {grid_search.best_params_['metric']}")
print(f"Best weighting type: {grid_search.best_params_['weights']}")

# Only "transform" the test data
X_test_scaled = scaler.transform(X_test)

# Making Predictions and Evaluation
best_knn = grid_search.best_estimator_

# Predict on the 3-second segments
y_pred_segments = best_knn.predict(X_test_scaled)

# Putting the results into a Pandas DataFrame for easy grouping
results_df = pd.DataFrame({
    'song_id': groups_test,
    'true_label': y_test,
    'pred_label': y_pred_segments
})

# Group by song_id, take the first true label, and find the mode 
# (most common) prediction
song_predictions = results_df.groupby('song_id').agg(
    true_genre=('true_label', 'first'),
    predicted_genre=('pred_label', lambda x: x.mode()[0])
)

final_accuracy = accuracy_score(song_predictions['true_genre'], song_predictions['predicted_genre'])

print(f"Best CV Accuracy (GroupKFold): {grid_search.best_score_ * 100:.2f}%")
print(f"Segment-level Accuracy: {accuracy_score(y_test, y_pred_segments) * 100:.2f}%")
print(f"Song-level Accuracy (Majority Vote): {final_accuracy * 100:.2f}%\n")

print("Detailed Classification Report (Song Level):")
print(classification_report(song_predictions['true_genre'], song_predictions['predicted_genre']))

# %%
# Feature plots

import matplotlib.pyplot as plt

plt.imshow((X[:, :n_mfcc].T), cmap='viridis', origin='lower', aspect=200)
plt.title("Averaged MFCCs")
plt.xlabel("Song number")
plt.ylabel("MFCC coefficient")
plt.show()
plt.imshow((X[:, n_mfcc:2*n_mfcc].T), cmap='viridis', origin='lower', aspect=200)
plt.title("Variance of MFCCs")
plt.xlabel("Song number")
plt.ylabel("MFCC coefficient")
plt.show()

# %%
# More visualizations

from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- 1. t-SNE PROJECTION ---
# Reduce dimensions to 2 with t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X_train_scaled)

df_tsne = pd.DataFrame(X_embedded, columns=['Component 1', 'Component 2'])
df_tsne['Genre'] = y_train

plt.figure(figsize=(10, 7))
sns.scatterplot(data=df_tsne, x='Component 1', y='Component 2', hue='Genre', palette='tab10', alpha=0.7)
plt.title("t-SNE Projection of Data Separation", fontsize=14, fontweight='bold')
plt.legend(title='Genres', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- 2. MFCC 1 vs MFCC 2 ---
mfcc_1 = X[:, 0]
mfcc_2 = X[:, 1]
plot_df_mfcc = pd.DataFrame({'MFCC 1': mfcc_1, 'MFCC 2': mfcc_2, 'Genre': y})

plt.figure(figsize=(10, 7))
sns.scatterplot(data=plot_df_mfcc, x='MFCC 1', y='MFCC 2', hue='Genre', palette='tab10', alpha=0.6, edgecolor='none')
plt.title('Data Separation via First Two MFCC Features', fontsize=14, fontweight='bold')
plt.xlabel('MFCC 1 (Overall Energy / Spectral Envelope)')
plt.ylabel('MFCC 2 (Bright vs. Dull / Spectral Slope)')
plt.legend(title='Genres', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- 3. VARIANCE 1 vs VARIANCE 2 ---
var_1 = X[:, n_mfcc]
var_2 = X[:, n_mfcc+1]
plot_df_var = pd.DataFrame({'Variance 1': var_1, 'Variance 2': var_2, 'Genre': y})

plt.figure(figsize=(10, 7))
sns.scatterplot(data=plot_df_var, x='Variance 1', y='Variance 2', hue='Genre', palette='tab10', alpha=0.6, edgecolor='none')
plt.title('Data Separation via First Two MFCC Variances', fontsize=14, fontweight='bold')
plt.xlabel('MFCC 1 (Overall Energy / Spectral Envelope Variance)')
plt.ylabel('MFCC 2 (Bright vs. Dull / Spectral Slope Variance)')
plt.legend(title='Genres', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- 4. CONFUSION MATRIX
plt.figure(figsize=(12, 10))
cm = confusion_matrix(song_predictions['true_genre'], song_predictions['predicted_genre'])

# We can get the unique display labels directly from the DataFrame
unique_genres = sorted(song_predictions['true_genre'].unique())

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_genres)
disp.plot(cmap='Blues', xticks_rotation='vertical', ax=plt.gca())

plt.title("kNN Genre Classification Confusion Matrix (Majority Vote)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %%
# Predict genres of new (imported) songs

from pathlib import Path
from collections import Counter

unknown_folder = Path.cwd() / "Data" / "unknown_music"

for file_name in os.listdir(unknown_folder):

    if not file_name.endswith(".wav"):
        continue

    file_path = unknown_folder / file_name

    try:
        audio, sample_rate = librosa.load(file_path, sr=None)

        segment_duration = 3
        samples_per_segment = int(segment_duration * sample_rate)

        feature_list = []

        # Extract features from every 3-second segment
        for i in range(0, len(audio), samples_per_segment):

            segment = audio[i:i + samples_per_segment]

            if len(segment) != samples_per_segment:
                continue

            # MFCC
            mfccs = librosa.feature.mfcc(
                y=segment,
                sr=sample_rate,
                n_mfcc=n_mfcc
            )
            mfccs_mean = np.mean(mfccs.T, axis=0)
            mfccs_var = np.var(mfccs.T, axis=0)

            # Chroma
            chroma = librosa.feature.chroma_stft(
                y=segment,
                sr=sample_rate
            )
            chroma_mean = np.mean(chroma.T, axis=0)
            chroma_var = np.var(chroma.T, axis=0)

            # Spectral centroid
            centroid = librosa.feature.spectral_centroid(
                y=segment
            )
            centroid_mean = np.mean(centroid.T, axis=0)
            centroid_var = np.var(centroid.T, axis=0)

            feature_vector = np.hstack((
                mfccs_mean,
                mfccs_var,
                chroma_mean,
                chroma_var,
                centroid_mean,
                centroid_var
            ))

            feature_list.append(feature_vector)

        if len(feature_list) == 0:
            print(f"{file_name}: No valid segments found.")
            continue

        # Scale features
        X_new = np.array(feature_list)
        X_new_scaled = scaler.transform(X_new)

        # Predict every segment
        segment_predictions = best_knn.predict(X_new_scaled)

        # Majority vote
        final_prediction = Counter(segment_predictions).most_common(1)[0][0]

        print(f"\nFile: {file_name}")
        print(f"Predicted genre: {final_prediction}")

        print("Segment predictions:")
        print(segment_predictions)

    except Exception as e:
        print(f"Error processing {file_name}: {e}")
