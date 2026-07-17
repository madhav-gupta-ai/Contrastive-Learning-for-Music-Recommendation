"""Collaborative filtering recommender (matrix factorization SVD).

Trains on the full filtered dataset: every (playlist, song) pair is a
positive interaction (rating 1) and, because the data is one-class, an equal
number of randomly sampled unobserved pairs per playlist is added with
rating 0 — without them the factorization would predict 1 for every pair.

What gets saved is the learned **item factor matrix** (one latent vector per
song, results/svd_factors.npz). Recommendations for an arbitrary playlist
then work like the other methods: average the factors of the playlist's
songs and rank candidates by cosine similarity to that mean — no user needs
to exist in the factorization.
"""

import os
import random

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader

from src import config
from src.create_triplets import build_playlists
from src.data_prep import load_audio_features


def user_item_interaction(playlists, start=0):
    """One row per (user, song) pair: the user has the song in their playlist."""
    playlist_data = []
    for user_id, playlist in enumerate(playlists, start=start):
        for song in playlist:
            # If a song occurs in a playlist, mark as 1 (liked)
            playlist_data.append([str(user_id), song, 1])

    return pd.DataFrame(playlist_data, columns=['user_id', 'song_id', 'rating'])


def sample_negatives(playlists, all_songs, start=0):
    """For each user, sample as many unobserved songs (rating 0) as they have
    positives, so the factorization sees both classes."""
    all_songs_set = set(all_songs)
    rows = []
    for user_id, playlist in enumerate(playlists, start=start):
        candidates = list(all_songs_set - set(playlist))
        negatives = random.sample(candidates, min(len(playlist), len(candidates)))
        for song in negatives:
            rows.append([str(user_id), song, 0])

    return pd.DataFrame(rows, columns=['user_id', 'song_id', 'rating'])


def rank_songs(playlist):
    """Rank all library songs by cosine similarity of their item factors to
    the mean item factor of the playlist's songs (best first)."""
    data = np.load(config.SVD_FACTORS_PATH)
    songs, factors = data['songs'], data['factors']

    index = {song: i for i, song in enumerate(songs)}
    playlist_idx = [index[song] for song in playlist if song in index]
    mean_factor = factors[playlist_idx].mean(axis=0)

    scores = (factors @ mean_factor) / (np.linalg.norm(factors, axis=1)
                                        * np.linalg.norm(mean_factor) + 1e-12)

    playlist_set = set(playlist)
    recommendations = [(song, score) for song, score in zip(songs, scores)
                       if song not in playlist_set]
    recommendations.sort(key=lambda x: x[1], reverse=True)

    return recommendations


def main():
    random.seed(config.RANDOM_SEED)

    df = pd.read_csv(config.FILTERED_DATASET_PATH)
    playlists = build_playlists(df)
    all_songs = load_audio_features()['unique_track'].tolist()
    print(f"Loaded {len(playlists)} playlists")

    interactions = pd.concat([user_item_interaction(playlists),
                              sample_negatives(playlists, all_songs)])

    # Importing the Reader and initializing it to read on a scale of 0 to 1
    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(interactions[['user_id', 'song_id', 'rating']], reader)
    trainset = data.build_full_trainset()

    print(f"Number of users: {trainset.n_users}")
    print(f"Number of items: {trainset.n_items}")
    print(f"Number of ratings: {trainset.n_ratings}")

    algo = SVD(random_state=config.RANDOM_SEED)
    algo.fit(trainset)

    songs = np.array([trainset.to_raw_iid(i) for i in range(trainset.n_items)])
    os.makedirs(os.path.dirname(config.SVD_FACTORS_PATH), exist_ok=True)
    np.savez_compressed(config.SVD_FACTORS_PATH,
                        songs=songs, factors=algo.qi.astype('float32'))
    print(f"Saved item factors for {len(songs)} songs: {config.SVD_FACTORS_PATH}")


if __name__ == '__main__':
    main()
