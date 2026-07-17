"""Content-based filtering recommender (KNN on audio features).

There is nothing to train — the "model" is the audio-feature table itself.
The 8 audio features of a playlist's songs are averaged into a centroid, and
candidate songs are ranked by Euclidean distance to it (closest first).
"""

import numpy as np

from src.data_prep import load_audio_features


def rank_songs(playlist, target_df=None):
    """Rank candidate songs by distance to the playlist's feature centroid.

    playlist:  list of song names present in data/audio_features.csv
    target_df: candidate songs to rank (name + feature columns);
               defaults to the whole English library
    """
    audio_features_df = load_audio_features()
    features_columns = audio_features_df.columns.drop(['unique_track', 'id'])
    if target_df is None:
        target_df = audio_features_df

    playlist_df = audio_features_df[audio_features_df['unique_track'].isin(playlist)]
    avg_attributes = playlist_df[features_columns].mean()

    # Exclude songs already in the playlist
    remaining_songs_df = target_df[~target_df['unique_track'].isin(playlist)]

    # Calculate Euclidean distances of all candidate songs to this average vector
    distances = np.linalg.norm(remaining_songs_df[features_columns] - avg_attributes, axis=1)

    # Combine song names and distances into a list of tuples and sort by distance
    song_distance_tuples = list(zip(remaining_songs_df['unique_track'], distances))
    song_distance_tuples.sort(key=lambda x: x[1])

    return song_distance_tuples
