"""Central configuration: paths, constants and hyperparameters.

All paths are relative to the repository root — run every script from there,
e.g. `python -m src.data_prep`.
"""

RANDOM_SEED = 42

# --- Paths ---------------------------------------------------------------
RAW_DATASET_PATH = 'data/spotify_dataset.csv'        # from Kaggle, see data/README.md
FULL_DATASET_PATH = 'data/full_dataset.csv'          # cleaned, unfiltered checkpoint
FILTERED_DATASET_PATH = 'data/filtered_dataset.csv'  # working dataset

AUDIO_FEATURES_PATH = 'data/audio_features.csv'      # standardized 8-feature vectors
AUDIO_FEATURES_RAW_PATH = 'data/audio_features_raw.csv'  # raw Spotify API payload
TRACK_IDS_PATH = 'data/track_csv.csv'                # Spotify track id <-> song name
HINDI_FEATURES_PATH = 'data/hindi_features.csv'      # global-recommendation demo songs

TRIPLETS_PATH = 'data/triplets.csv'

MODEL_PATH = 'results/model.pth'
TRAIN_LOSS_PATH = 'results/train_loss.csv'
SVD_FACTORS_PATH = 'results/svd_factors.npz'

# --- Dataset filtering ----------------------------------------------------
MIN_SONG_COUNT = 50        # a song must appear in at least this many playlists
MIN_PLAYLIST_SIZE = 50     # unique songs per playlist
MAX_PLAYLIST_SIZE = 2500

# --- Triplet mining -------------------------------------------------------
TRIPLETS_A = 10            # minimum (and multiplier of) triplets per song
TRIPLETS_B = 100           # popularity scaling: count = round(freq / (n_playlists / b)) * a
SIMILARITY_LIST_SIZE = 200 # most/least similar songs kept per anchor

# --- Contrastive model (Optuna-tuned values from the ICE 2025 paper) -------
LAYER_SIZES = [256, 128, 64, 32, 32, 16]  # last entry is the embedding dimension
LEARNING_RATE = 0.0023654517608398635
BATCH_SIZE = 64
NUM_EPOCHS = 10
MARGIN = 3.157321757004013
