"""Triplet mining from the playlists of the filtered dataset.

Number of triplets to be made for a song
    = [ no. of song occurrences / (total no. of playlists / b) ] * a
Minimum no. of triplets for a song = a

Similarity of song B to song A:
    sim(A, B) = (1 + no. of playlists in which both A and B occur)
                / total no. of playlists in which B occurs

For each anchor song, positives are drawn from its most similar songs and
negatives from its least similar songs. Output: data/triplets.csv with
columns (anchor, positive, negative).
"""

from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from src import config


def build_playlists(df):
    """Group the dataset into one deduplicated song list per user/playlist."""
    return list(df.groupby('user_id')['unique_track'].apply(lambda s: list(pd.unique(s))))


def triplet_budget(playlists):
    """Popularity-weighted number of triplets per song (minimum TRIPLETS_A)."""
    track_counts = pd.Series([track for playlist in playlists for track in playlist]).value_counts()
    triplets_count = round(track_counts / (len(playlists) / config.TRIPLETS_B), 0)
    triplets_count = triplets_count.apply(lambda x: max(x, 1) * config.TRIPLETS_A)
    return triplets_count


def similarity_lists(playlists):
    """Per song: the SIMILARITY_LIST_SIZE most and least similar songs."""
    # Creating a dictionary of each song to the set of playlist indexes it appears in
    track_playlists = defaultdict(set)
    for i, playlist in enumerate(playlists):
        for track in playlist:
            track_playlists[track].add(i)

    # Creating a song-similarity matrix and calculating the similarity between each pair of songs
    similarity_scores = defaultdict(lambda: defaultdict(float))
    for track_a in tqdm(track_playlists, desc="Computing similarities"):
        for track_b in track_playlists:
            if track_a != track_b:
                common_playlists = track_playlists[track_a].intersection(track_playlists[track_b])
                score = (1 + len(common_playlists)) / len(track_playlists[track_b])
                similarity_scores[track_a][track_b] = score

    n = config.SIMILARITY_LIST_SIZE
    most_similar_tracks = {track: sorted(similarity_scores[track].items(), key=lambda x: x[1], reverse=True)[:n]
                           for track in similarity_scores}
    least_similar_tracks = {track: sorted(similarity_scores[track].items(), key=lambda x: x[1])[:n]
                            for track in similarity_scores}
    return most_similar_tracks, least_similar_tracks


def build_triplets(triplets_count, most_similar_tracks, least_similar_tracks):
    triplets_count = pd.DataFrame(triplets_count).reset_index()
    triplets_count.columns = ["track_name", "triplet_count"]

    rows_list = []
    for index, row in tqdm(triplets_count.iterrows(), total=triplets_count.shape[0],
                           desc="Processing triplets"):
        track_name = row['track_name']
        triplet_count = int(row['triplet_count'])

        most_similar = most_similar_tracks[track_name][:triplet_count] if track_name in most_similar_tracks else []
        least_similar = least_similar_tracks[track_name][:triplet_count] if track_name in least_similar_tracks else []
        least_similar = sorted(least_similar, key=lambda x: x[1], reverse=True)

        for i in range(triplet_count):
            if i < len(most_similar) and i < len(least_similar):
                rows_list.append({"anchor": track_name,
                                  "positive": most_similar[i][0],
                                  "negative": least_similar[i][0]})

    return pd.DataFrame(rows_list, columns=['anchor', 'positive', 'negative'])


def main():
    df = pd.read_csv(config.FILTERED_DATASET_PATH)
    playlists = build_playlists(df)
    print(f"Loaded {len(playlists)} playlists")

    triplets_count = triplet_budget(playlists)
    print(f"Total number of triplets: {int(triplets_count.sum())}")

    most_similar_tracks, least_similar_tracks = similarity_lists(playlists)

    triplets_df = build_triplets(triplets_count, most_similar_tracks, least_similar_tracks)
    print(triplets_df)
    triplets_df.to_csv(config.TRIPLETS_PATH, index=False)
    print(f"Saved: {config.TRIPLETS_PATH}")


if __name__ == '__main__':
    main()
