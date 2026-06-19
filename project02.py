# %%
# Data import function

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
                #Take the variance of each MFCC coefficient
                mfccs_var = np.var(mfccs, axis=1)
                #Combine features (horizontally for classifier)
                feature_vector = np.hstack((mfccs_mean, mfccs_var))
                # Extract genre from filename
                # Example: blues00000.wav -> blues
                genre = ''.join([char for char in file_name if char.isalpha()])
                genre = genre.rstrip("wav")

                # Store data
                features.append(feature_vector)
                labels.append(genre)

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)

    return X, y

# %%
# Data reading & proccessing

from pathlib import Path

folder_path = Path.cwd() / "Data" / "genres_original"
n_mfcc=20 #Feature number ~ hyperparameter
X, y = load_music_data(folder_path, n_mfcc) 
print(len(y))
print(np.shape(X))

# %%
# Data splitting

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=67, 
    stratify=y
)

# %%
# Initial kNN model

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

scaler = StandardScaler()

# "Fit" learns the mean/variance of the training data, "transform" applies it
X_train_scaled = scaler.fit_transform(X_train)

# This searches for the best parameters number using cv-fold cross validation - Hyperparameter?
param_grid = {
    'n_neighbors': 3 + 2*np.arange(25),
    'metric': ['euclidean', 'minkowski', 'cosine', 'hamming'],
    'weights': ['uniform', 'distance']
    }

grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=40, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

print(f"Best K Value: {grid_search.best_params_['n_neighbors']}")
print(f"Best distance type: {grid_search.best_params_['metric']}")
print(f"Best weighting type: {grid_search.best_params_['weights']}")
print(f"Best CV Accuracy: {grid_search.best_score_ * 100:.2f}%")

# ONLY "transform" the test data. Never "fit" on test data (that is cheating!)
X_test_scaled = scaler.transform(X_test)

# Making Predictions and Evaluation
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test_scaled)

# Print out the results
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")

# The classification report shows exactly which genres the model is struggling with
print("Detailed Classification Report:")
print(classification_report(y_test, y_pred))

# %%
# Feature plots

import matplotlib.pyplot as plt

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
# More visualizations

from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Reduce dimensions to 2 with t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X_train_scaled)
df_tsne = pd.DataFrame(X_embedded, columns=['Component 1', 'Component 2'])
df_tsne['Genre'] = y_train
sns.scatterplot(data=df_tsne, x='Component 1', y='Component 2', hue='Genre', palette='tab10', alpha=0.7)
plt.title("t-SNE Projection of Data Separation")

# 2D with first two mfccs
mfcc_1 = X[:, 0]
mfcc_2 = X[:, 1]
plot_df = pd.DataFrame({
    'MFCC 1': mfcc_1,
    'MFCC 2': mfcc_2,
    'Genre': y
})
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=plot_df, 
    x='MFCC 1', 
    y='MFCC 2', 
    hue='Genre', 
    palette='tab10',  # Color palette
    alpha=0.6,        # Transparency
    edgecolor='none'
)
plt.title('Data Separation via First Two MFCC Features', fontsize=14, fontweight='bold')
plt.xlabel('MFCC 1 (Overall Energy / Spectral Envelope)')
plt.ylabel('MFCC 2 (Bright vs. Dull / Spectral Slope)')
plt.legend(title='Genres', bbox_to_anchor=(1.05, 1), loc='upper left') # Moves legend outside
plt.tight_layout()
plt.show()

# 2D with first two variances
var_1 = X[:, n_mfcc]
var_2 = X[:, n_mfcc+1]
plot_df = pd.DataFrame({
    'Variance 1': var_1,
    'Variance 2': var_2,
    'Genre': y
})
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=plot_df, 
    x='Variance 1', 
    y='Variance 2', 
    hue='Genre', 
    palette='tab10',  # Color palette
    alpha=0.6,        # Transparency
    edgecolor='none'
)
plt.title('Data Separation via First Two MFCC Variances', fontsize=14, fontweight='bold')
plt.xlabel('MFCC 1 (Overall Energy / Spectral Envelope Variance)')
plt.ylabel('MFCC 2 (Bright vs. Dull / Spectral Slope Variance)')
plt.legend(title='Genres', bbox_to_anchor=(1.05, 1), loc='upper left') # Moves legend outside
plt.tight_layout()
plt.show()

# %%
