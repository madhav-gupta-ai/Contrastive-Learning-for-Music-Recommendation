"""Fetch Spotify audio features for every song in the filtered dataset.

NOTE: Spotify deprecated the `audio-features` endpoint for new API apps in
November 2024, so this script may not work with newly created credentials.
The fetched data ships with this repository (data/audio_features.csv and
data/audio_features_raw.csv); this script is kept for provenance and for
apps that still have access.

Credentials are read from the environment variables SPOTIFY_CLIENT_ID and
SPOTIFY_CLIENT_SECRET.

Pipeline: song names -> track ids (search, with retries and exponential
backoff) -> audio features (batched) -> standardized 8-feature vectors.
"""

import os
import random
import time

import pandas as pd
import requests
import spotipy
from sklearn.preprocessing import StandardScaler
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm

from src import config

client_id = os.environ.get('SPOTIFY_CLIENT_ID')
client_secret = os.environ.get('SPOTIFY_CLIENT_SECRET')


def get_spotify_client():
    if not client_id or not client_secret:
        raise SystemExit("Set the SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET "
                         "environment variables to use the Spotify API.")
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id,
                                                          client_secret=client_secret)
    return spotipy.Spotify(client_credentials_manager=client_credentials_manager,
                           requests_timeout=30)


def exponential_backoff(retry):
    """Calculate sleep time for exponential backoff strategy."""
    return min(60, (2 ** retry) + random.uniform(0, 1))


def get_track_ids(song_names, retries=3):
    sp = get_spotify_client()

    track_ids = []
    songs_not_found = []
    for song_name in tqdm(song_names, desc="Fetching track IDs"):
        for attempt in range(retries):
            try:
                result = sp.search(q=song_name, limit=1)
                if result['tracks']['items']:
                    track_id = result['tracks']['items'][0]['id']
                    track_ids.append((track_id, song_name))  # Store both track ID and song fullname
                else:
                    songs_not_found.append(song_name)
                break  # Break out of the retry loop on success
            except (spotipy.SpotifyException, requests.exceptions.RequestException, Exception) as e:
                print(f"Error fetching track ID for '{song_name}': {e}. Attempt {attempt + 1} of {retries}")
                time.sleep(exponential_backoff(attempt))  # Exponential backoff
                if attempt == retries - 1:
                    print(f"Failed to fetch track ID for '{song_name}' after {retries} attempts.")
                    songs_not_found.append(song_name)
    return track_ids, songs_not_found


def get_audio_features_batch(track_ids_with_names):
    """Batch request audio features, including the 'unique_track' column."""
    sp = get_spotify_client()

    batch_size = 100
    features_list = []
    tracks_missing_features = []
    for i in tqdm(range(0, len(track_ids_with_names), batch_size), desc="Fetching audio features"):
        batch = track_ids_with_names[i:i + batch_size]
        track_ids = [x[0] for x in batch]  # Extract just the track IDs for the API call
        features_results = sp.audio_features(track_ids)
        for feature, (track_id, song_name) in zip(features_results, batch):
            if feature:
                feature['unique_track'] = song_name  # Add the 'unique_track' field
                features_list.append(feature)
            else:
                tracks_missing_features.append(song_name)

    return features_list, tracks_missing_features


def process_songs_in_batches(song_names, batch_size=1000):
    track_ids = []
    songs_not_found = []

    for i in range(0, len(song_names), batch_size):
        batch_song_names = song_names[i:i + batch_size]
        batch_track_ids, batch_songs_not_found = get_track_ids(batch_song_names)
        track_ids.extend(batch_track_ids)
        songs_not_found.extend(batch_songs_not_found)

    return track_ids, songs_not_found


def standardize_features():
    """Reduce the raw API payload to 8 z-scored features per song."""
    df = pd.read_csv(config.AUDIO_FEATURES_RAW_PATH)

    df = df[df['unique_track'].notna()]  # Drop rows where 'unique_track' is NaN
    df = df.drop_duplicates(subset='unique_track')
    df.drop(['key', 'mode', 'tempo', 'type', 'uri', 'track_href', 'analysis_url',
             'duration_ms', 'time_signature'], axis=1, inplace=True)

    # Normalize the remaining columns (excluding 'unique_track' and 'id')
    features_columns = df.columns.drop(['unique_track', 'id'])
    scaler = StandardScaler()
    df[features_columns] = scaler.fit_transform(df[features_columns])

    df.to_csv(config.AUDIO_FEATURES_PATH, index=False)
    print(f"Saved standardized features for {len(df)} songs: {config.AUDIO_FEATURES_PATH}")


def main():
    song_names = pd.read_csv(config.FILTERED_DATASET_PATH)['unique_track'].unique().tolist()
    print(f"Fetching features for {len(song_names)} songs")

    track_ids, songs_not_found = process_songs_in_batches(song_names)
    pd.DataFrame(track_ids).to_csv(config.TRACK_IDS_PATH, index=False)

    audio_features, tracks_missing_features = get_audio_features_batch(track_ids)

    # Retry the songs that failed the first pass
    songs_not_found = songs_not_found + tracks_missing_features
    missing_song_ids, still_missing = process_songs_in_batches(songs_not_found)
    missing_song_features, more_missing = get_audio_features_batch(missing_song_ids)
    all_missing = more_missing + still_missing

    audio_features_df = pd.concat([pd.DataFrame(audio_features),
                                   pd.DataFrame(missing_song_features)], ignore_index=True)
    audio_features_df.to_csv(config.AUDIO_FEATURES_RAW_PATH, index=False)
    print(f"Final audio features saved. Total: {len(audio_features_df)}")
    print(f"Tracks not found: {len(still_missing)}, Tracks missing features: {len(more_missing)}, "
          f"Total missing: {len(all_missing)}")

    standardize_features()


if __name__ == '__main__':
    main()
