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
                        
                        # 2. Delta & Delta-Delta MFCCs (Temporal change of Timbre)
                        delta_mfccs = librosa.feature.delta(mfccs)
                        delta_mfccs_mean = np.mean(delta_mfccs.T, axis=0)
                        delta_mfccs_var = np.var(delta_mfccs.T, axis=0)
                        
                        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
                        delta2_mfccs_mean = np.mean(delta2_mfccs.T, axis=0)
                        delta2_mfccs_var = np.var(delta2_mfccs.T, axis=0)
                        
                        # 3. Chroma STFT (Harmony / Pitch Classes)
                        chroma = librosa.feature.chroma_stft(y=segment, sr=sample_rate)
                        chroma_mean = np.mean(chroma.T, axis=0)
                        chroma_var = np.var(chroma.T, axis=0)
                        
                        # 4. Spectral Centroid (Brightness of Sound)
                        centroid = librosa.feature.spectral_centroid(y=segment, sr=sample_rate)
                        centroid_mean = np.mean(centroid.T, axis=0)
                        centroid_var = np.var(centroid.T, axis=0)
                        
                        # 5. Tempo / BPM (Rhythm)
                        tempo_data, _ = librosa.beat.beat_track(y=segment, sr=sample_rate)
                        # Wrap in np.array to ensure it stacks cleanly horizontally
                        tempo = np.array([np.mean(tempo_data)])

                        # Combine all features into one row
                        feature_vector = np.hstack((
                            mfccs_mean, 
                            mfccs_var,
                            delta_mfccs_mean, 
                            delta_mfccs_var,
                            delta2_mfccs_mean, 
                            delta2_mfccs_var,
                            chroma_mean, 
                            chroma_var,
                            centroid_mean, 
                            centroid_var,
                            tempo
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
n_mfcc=20 #Mfcc feature number ~ hyperparameter
X, y, groups = load_music_data(folder_path, n_mfcc) 
print(len(y))
print(np.shape(X))

# %%
# Data splitting

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GroupShuffleSplit # For protection against
# target leakage

# Split the data, keeping segments of the same song together
gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=20)

# gss.split yields indices for train and test
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
groups_train, groups_test = groups[train_idx], groups[test_idx]

# %%
# Model Training and Evaluation (kNN & Random Forest)

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import accuracy_score, classification_report

# 1. Define the GroupKFold cross-validation strategy
gkf = GroupKFold(n_splits=5)

# =====================================================================
# K-NEAREST NEIGHBORS (kNN)
# =====================================================================
print("--- Training K-Nearest Neighbors ---")

knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

knn_param_grid = {
    'knn__n_neighbors': 3 + 2 * np.arange(15),
    'knn__metric': ['euclidean', 'minkowski', 'cosine', 'manhattan'],
    'knn__weights': ['uniform', 'distance']
}

knn_grid_search = GridSearchCV(knn_pipeline, knn_param_grid, cv=gkf, scoring='accuracy')
knn_grid_search.fit(X_train, y_train, groups=groups_train)

print(f"Best K Value: {knn_grid_search.best_params_['knn__n_neighbors']}")
print(f"Best distance type: {knn_grid_search.best_params_['knn__metric']}")
print(f"Best weighting type: {knn_grid_search.best_params_['knn__weights']}\n")


# =====================================================================
# RANDOM FOREST (RF)
# =====================================================================
print("--- Training Random Forest ---")

# Tree-based models don't strictly require scaling, but a pipeline 
# keeps the structure clean and consistent.
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('rf', RandomForestClassifier(random_state=1))
])

rf_param_grid = {
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [None, 10, 20],
    'rf__min_samples_split': [2, 5, 10]
}

rf_grid_search = GridSearchCV(rf_pipeline, rf_param_grid, cv=gkf, scoring='accuracy')
rf_grid_search.fit(X_train, y_train, groups=groups_train)

print(f"Best Estimators: {rf_grid_search.best_params_['rf__n_estimators']}")
print(f"Best Max Depth: {rf_grid_search.best_params_['rf__max_depth']}")
print(f"Best Min Samples Split: {rf_grid_search.best_params_['rf__min_samples_split']}\n")


# =====================================================================
# EVALUATION FUNCTION
# =====================================================================
def evaluate_model(model_name, grid_search_object):
    """
    Evaluates the testing on both models

    Parameters: Name of model (string), grid search object
    """
    
    best_model = grid_search_object.best_estimator_
    
    # Predict on the 3-second segments
    y_pred_segments = best_model.predict(X_test)
    
    # Organize results into a DataFrame for song-level voting
    results_df = pd.DataFrame({
        'song_id': groups_test,
        'true_label': y_test,
        'pred_label': y_pred_segments
    })
    
    # Group by song_id, take the first true label, and find the mode prediction
    song_predictions = results_df.groupby('song_id').agg(
        true_genre=('true_label', 'first'),
        predicted_genre=('pred_label', lambda x: x.mode()[0])
    )
    
    final_accuracy = accuracy_score(song_predictions['true_genre'], song_predictions['predicted_genre'])
    
    print(f"===== {model_name} Results =====")
    print(f"Best CV Accuracy (GroupKFold): {grid_search_object.best_score_ * 100:.2f}%")
    print(f"Segment-level Accuracy: {accuracy_score(y_test, y_pred_segments) * 100:.2f}%")
    print(f"Song-level Accuracy (Majority Vote): {final_accuracy * 100:.2f}%\n")
    
    print(f"Detailed Classification Report ({model_name} Song Level):")
    print(classification_report(song_predictions['true_genre'], song_predictions['predicted_genre']))
    print("\n" + "="*50 + "\n")
    return song_predictions

# Run evaluation for both models
knn_song_predictions = evaluate_model("kNN", knn_grid_search)
rf_song_predictions = evaluate_model("Random Forest", rf_grid_search)

# %%
# Accuracy vs. k plot

import matplotlib.pyplot as plt

results = pd.DataFrame(knn_grid_search.cv_results_)

best_metric = knn_grid_search.best_params_['knn__metric']
best_weight = knn_grid_search.best_params_['knn__weights']

plot_data = results[
    (results['param_knn__metric'] == best_metric) &
    (results['param_knn__weights'] == best_weight)
].sort_values('param_knn__n_neighbors')

plt.figure(figsize=(8,5))

plt.plot(
    plot_data['param_knn__n_neighbors'], # FIX
    plot_data['mean_test_score'] * 100,
    marker='o',
    linewidth=2
)

plt.scatter(
    knn_grid_search.best_params_['knn__n_neighbors'], # FIX
    knn_grid_search.best_score_ * 100,
    s=120,
    label="Best k"
)

plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Cross-Validation Accuracy (%)")
plt.title(f"Accuracy vs. k ({best_metric}, {best_weight})")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.show()

# %%
# Distance metric comparison

results = pd.DataFrame(knn_grid_search.cv_results_)

metrics = knn_param_grid['knn__metric']
metric_scores = []

# Find the best score achieved by each distance metric
for metric in metrics:
    best_score = results[
        results['param_knn__metric'] == metric
    ]['mean_test_score'].max()

    metric_scores.append(best_score * 100)

plt.figure(figsize=(7,5))

plt.bar(metrics, metric_scores)

plt.ylabel("Best Cross-Validation Accuracy (%)")
plt.xlabel("Distance Metric")
plt.title("Comparison of Distance Metrics")

# Add value labels
for i, score in enumerate(metric_scores):
    plt.text(i, score + 0.3, f"{score:.2f}%", ha='center')

plt.ylim(0, max(metric_scores) + 5)

plt.tight_layout()
plt.show()

# %%
# Feature plots with sorted genres

import matplotlib.pyplot as plt
import numpy as np

# Sorting data by genre
sort_idx = np.argsort(y)
X_sorted = X[sort_idx]
y_sorted = y[sort_idx]

# Finding the transition indices where genres change
transition_indices = np.flatnonzero(y_sorted[:-1] != y_sorted[1:]) + 1

# Getting the genre names at each block for labeling
unique_genres, unique_indices = np.unique(y_sorted, return_index=True)
genre_order = [y_sorted[idx] for idx in sorted(unique_indices)]

# Calculating the middle point of each genre block to center the text labels
block_starts = np.insert(transition_indices, 0, 0)
block_ends = np.append(transition_indices, len(y_sorted))
label_positions = (block_starts + block_ends) / 2

# =====================================================================
# --- PLOT 1: AVERAGED MFCCs ---
# =====================================================================
fig1, ax1 = plt.subplots(figsize=(12, 5))

# Skipping MFCC 0 / loudness
im1 = ax1.imshow(X_sorted[:, 1:n_mfcc].T, cmap='viridis', origin='lower', aspect=500,
                vmin = 0, vmax = 100)
ax1.set_title("Averaged MFCCs (Grouped by Genre)\n\n", fontsize=14, weight='bold')
ax1.set_xlabel("Segment number")
ax1.set_ylabel("MFCC coefficient")
ax1.set_ylim(1, n_mfcc-2)
fig1.colorbar(im1, ax=ax1, shrink=0.8, label="Magnitude")

# Drawing vertical lines at transitions
for x_pos in transition_indices:
    ax1.axvline(x=x_pos, color='white', linestyle='--', alpha=0.7, linewidth=1.5)

# Placing labels above the plot area
for label, pos in zip(genre_order, label_positions):
    ax1.text(pos, 1.02, label, color='black', weight='bold', 
             ha='center', va='bottom', rotation=30, fontsize=10,
             transform=ax1.get_xaxis_transform())

plt.tight_layout()
plt.show()

# =====================================================================
# --- PLOT 2: VARIANCE OF MFCCs ---
# =====================================================================
fig2, ax2 = plt.subplots(figsize=(12, 5))

im2 = ax2.imshow(X_sorted[:, n_mfcc+1 : 2*n_mfcc].T, cmap='viridis', origin='lower', aspect=500,
                vmin = 0, vmax = 1000)
ax2.set_title("Variance of MFCCs (Grouped by Genre)\n\n", fontsize=14, weight='bold')
ax2.set_xlabel("Segment number")
ax2.set_ylabel("MFCC coefficient")
ax2.set_ylim(1, n_mfcc-2)
fig2.colorbar(im2, ax=ax2, shrink=0.8, label="Variance")

for x_pos in transition_indices:
    ax2.axvline(x=x_pos, color='white', linestyle='--', alpha=0.7, linewidth=1.5)

for label, pos in zip(genre_order, label_positions):
    ax2.text(pos, 1.02, label, color='black', weight='bold', 
             ha='center', va='bottom', rotation=30, fontsize=10,
             transform=ax2.get_xaxis_transform())

plt.tight_layout()
plt.show()

#%%
#Other feature plots

# =====================================================================
# BPM
# =====================================================================
fig1, ax1 = plt.subplots(figsize=(12, 5))

im1 = ax1.plot(X_sorted[:, -1].T)
ax1.set_title("BMP by song\n\n", fontsize=14, weight='bold')
ax1.set_xlabel("Song number")
ax1.set_ylabel("BPM")
ax1.set_xlim(0, 1000)

# Drawing vertical lines at transitions
for x_pos in transition_indices:
    ax1.axvline(x=x_pos, color='black', linestyle='--', alpha=0.7, linewidth=1.5)

# Placing labels above the plot area
for label, pos in zip(genre_order, label_positions):
    ax1.text(pos, 1.02, label, color='black', weight='bold', 
             ha='center', va='bottom', rotation=30, fontsize=10,
             transform=ax1.get_xaxis_transform())

plt.tight_layout()
plt.show()

# =====================================================================
# Average tone (ONLY works with 990 datapoints (full snippets))
# =====================================================================
fig1, ax1 = plt.subplots(figsize=(12, 5))

x_avg = X_sorted.reshape(10, 99, 147).mean(axis=1)
im1 = ax1.imshow(x_avg[:, 6*n_mfcc:6*n_mfcc+12].T, cmap='viridis', origin='lower',
                vmin = 0.2, vmax = 0.5)
ax1.set_title("Averaged MFCCs (Grouped by Genre)\n\n", fontsize=14, weight='bold')
ax1.set_xlabel("Segment number")
ax1.set_ylabel("MFCC coefficient")
ax1.set_ylim(1, 11.5)
fig1.colorbar(im1, ax=ax1, shrink=0.8, label="Magnitude")

plt.tight_layout()
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
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
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
cm = confusion_matrix(knn_song_predictions['true_genre'], knn_song_predictions['predicted_genre'])

# We can get the unique display labels directly from the DataFrame
unique_genres = sorted(knn_song_predictions['true_genre'].unique())

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_genres)
disp.plot(cmap='Blues', xticks_rotation='vertical', ax=plt.gca())

plt.title("kNN Genre Classification Confusion Matrix (Majority Vote)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %%
# Feature comparison experiment

import itertools
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Define the exact sizes based on the extraction block
mfcc_size = 2 * n_mfcc           # mean + var
delta_size = 2 * n_mfcc          # mean + var
delta2_size = 2 * n_mfcc         # mean + var
chroma_size = 24                 # 12 mean + 12 var
centroid_size = 2                # 1 mean + 1 var
tempo_size = 1                   # 1 mean

feature_sizes = {
    "MFCC": mfcc_size,
    "Delta": delta_size,
    "Delta2": delta2_size,
    "Chroma": chroma_size,
    "Centroid": centroid_size,
    "Tempo": tempo_size
}

# Map column indices for each feature block sequentially
feature_indices = {}
current_idx = 0
for name, size in feature_sizes.items():
    feature_indices[name] = list(range(current_idx, current_idx + size))
    current_idx += size

# Generate all possible combinations
feature_sets = {}
feature_names = list(feature_sizes.keys())

for r in range(1, len(feature_names) + 1):
    for combo in itertools.combinations(feature_names, r):
        combo_name = " + ".join(combo)
        combined_indices = []
        for name in combo:
            combined_indices.extend(feature_indices[name])
        feature_sets[combo_name] = combined_indices

# Comparison loop
comparison_results = {}

print(f"\n========== FEATURE COMPARISON ({len(feature_sets)} combinations) ==========\n")

for name, cols in feature_sets.items():

    # Select the desired features
    X_train_subset = X_train[:, cols]
    X_test_subset = X_test[:, cols]

    # Scale them
    scaler_subset = StandardScaler()
    X_train_scaled_subset = scaler_subset.fit_transform(X_train_subset)
    X_test_scaled_subset = scaler_subset.transform(X_test_subset)

    # Train using the same best parameters found during GridSearch
    knn = KNeighborsClassifier(
        n_neighbors=knn_grid_search.best_params_["knn__n_neighbors"],
        metric=knn_grid_search.best_params_["knn__metric"],
        weights=knn_grid_search.best_params_["knn__weights"]
    )

    knn.fit(X_train_scaled_subset, y_train)

    # Predict segment labels
    y_pred = knn.predict(X_test_scaled_subset)

    # Majority vote per song
    temp_df = pd.DataFrame({
        "song_id": groups_test,
        "true": y_test,
        "pred": y_pred
    })

    song_results = temp_df.groupby("song_id").agg(
        true=("true", "first"),
        pred=("pred", lambda x: x.mode()[0])
    )

    accuracy = accuracy_score(song_results["true"], song_results["pred"])
    comparison_results[name] = accuracy

# Display the Top 10 Best Performing Combinations
print("\n--- TOP 10 FEATURE COMBINATIONS ---")

# Sort the dictionary by accuracy in descending order
sorted_results = sorted(comparison_results.items(), key=lambda item: item[1], reverse=True)

for i, (name, acc) in enumerate(sorted_results[:10]):
    print(f"{i+1}. {name:50s} : {acc*100:.2f}%")

# Display the Top 10 Best Performing Combinations
print("\n--- BOTTOM 10 FEATURE COMBINATIONS ---")

# Sort the dictionary by accuracy in descending order
sorted_results = sorted(comparison_results.items(), key=lambda item: item[1], reverse=True)

for i, (name, acc) in enumerate(sorted_results[-10:]):
    print(f"{i+54}. {name:50s} : {acc*100:.2f}%")

# %%
#Single song visualization

# 1. Scale the entire dataset
scaler_full = StandardScaler()
X_scaled_full = scaler_full.fit_transform(X)

# 2. Fit t-SNE on the entire dataset
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X_scaled_full)

# 3. Choose one song to highlight
# (Picking the very first song in the groups array as the target)
target_song_id = groups[0] 
target_genre = y[0]

# 4. Package data into a DataFrame for easy plotting
df_tsne = pd.DataFrame(X_embedded, columns=['Component 1', 'Component 2'])
df_tsne['Song ID'] = groups
df_tsne['Is Target'] = df_tsne['Song ID'] == target_song_id

# 5. Plot the results
plt.figure(figsize=(10, 7))

# Plot background points (all other songs)
sns.scatterplot(
    data=df_tsne[~df_tsne['Is Target']], 
    x='Component 1', 
    y='Component 2', 
    color='lightgray', 
    alpha=0.4, 
    edgecolor='none',
    label='Other Songs'
)

# Plot the 10 segments of the target song
sns.scatterplot(
    data=df_tsne[df_tsne['Is Target']], 
    x='Component 1', 
    y='Component 2', 
    color='red', 
    s=120,          # Increase marker size
    edgecolor='black',
    linewidth=1.5,  # Add a distinct border
    label=f'Target: {target_song_id} ({target_genre})'
)

plt.title(f"t-SNE: Visualizing Segment Clustering for a Single Song", fontsize=14, fontweight='bold')
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
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

        # Predict every segment
        segment_predictions = knn_grid_search.predict(X_new)

        # Majority vote
        final_prediction = Counter(segment_predictions).most_common(1)[0][0]

        print(f"\nFile: {file_name}")
        print(f"Predicted genre: {final_prediction}")

        print("Segment predictions:")
        print(segment_predictions)

    except Exception as e:
        print(f"Error processing {file_name}: {e}")


# %%
