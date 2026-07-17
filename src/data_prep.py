"""Data preparation: raw Kaggle CSV -> full_dataset.csv -> filtered_dataset.csv.

Steps:
1. Read the raw "Spotify Playlists" dataset (bad lines skipped).
2. Re-key playlists: username -> username + playlistname, so every playlist
   is treated as one independent "user".
3. Create `unique_track` = "artistname - trackname" to disambiguate songs.
4. Iteratively remove rare songs and out-of-range playlists until equilibrium.
5. Remove songs whose audio features are unavailable, re-run the filter.
"""

import pandas as pd

from src import config


def equilibrium_filter(df):
    """Iteratively remove songs below MIN_SONG_COUNT and playlists outside the
    size range until neither filter changes the dataset (a fixed point)."""
    counter = 1
    temp = 0
    new_temp = df['unique_track'].nunique()

    print(f"Initial Number of Songs: {new_temp}, Users: {df['user_id'].nunique()}")

    while temp != new_temp:
        print(f"Epoch {counter}")
        temp = new_temp

        # Removing rows where the song doesnt appear the minimum amount of times
        track_counts = df['unique_track'].value_counts()
        tracks_to_keep = track_counts[track_counts >= config.MIN_SONG_COUNT].index
        df = df[df['unique_track'].isin(tracks_to_keep)]

        # Removing users with playlist sizes out of range (less than min or more than max)
        user_track_counts = df.groupby('user_id')['unique_track'].nunique()
        users_to_keep = user_track_counts[
            (user_track_counts >= config.MIN_PLAYLIST_SIZE)
            & (user_track_counts <= config.MAX_PLAYLIST_SIZE)
        ].index
        df = df[df['user_id'].isin(users_to_keep)]

        new_temp = df['unique_track'].nunique()
        print(f"Current Number of Songs: {new_temp}, Users: {df['user_id'].nunique()}")
        counter += 1

    print("Equilibrium Reached")
    return df


def load_audio_features():
    """Load the standardized audio features, deduplicated on song name."""
    audio_features_df = pd.read_csv(config.AUDIO_FEATURES_PATH)
    return audio_features_df.drop_duplicates(subset='unique_track')


def main():
    # Reading the original RAW data file from Kaggle. Its header row contains
    # spaces after the commas ('"user_id", "artistname", ...'), so
    # skipinitialspace is needed for the column names to parse cleanly.
    print(f"Reading raw dataset from {config.RAW_DATASET_PATH}")
    df = pd.read_csv(config.RAW_DATASET_PATH, quotechar='"', escapechar='\\',
                     on_bad_lines='skip', header=0, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    # Changing username to username + playlist name such that each playlist is
    # treated as a separate user
    df['user_id'] = df['user_id'] + ' - ' + df['playlistname']

    # Creating a new column called 'unique_track' to deal with same trackname
    # but different artistname situations
    df['unique_track'] = df['artistname'] + ' - ' + df['trackname']

    df.to_csv(config.FULL_DATASET_PATH, index=False)
    print(f"Saved checkpoint: {config.FULL_DATASET_PATH}")

    df = equilibrium_filter(df)

    # Removing the tracks for which audio features could not be found, then
    # re-running the filter since removals may push playlists below the minimum
    audio_features_df = load_audio_features()
    df = df[df['unique_track'].isin(audio_features_df['unique_track'])]
    df = equilibrium_filter(df)

    df.to_csv(config.FILTERED_DATASET_PATH, index=False)
    print(f"Saved working dataset: {config.FILTERED_DATASET_PATH}")
    print(f"Final: {df['unique_track'].nunique()} songs, {df['user_id'].nunique()} playlists")


if __name__ == '__main__':
    main()
