"""Get recommendations for any playlist from the command line.

Examples (run from the repository root):

    python -m src.recommend --songs "Coldplay - Fix You" "Adele - Someone Like You"
    python -m src.recommend --method knn --top 10 --songs "Daft Punk - Get Lucky"
    python -m src.recommend --method svd --playlist-file my_playlist.txt
    python -m src.recommend --library hindi --songs "Coldplay - Fix You"

Song names use the "artistname - trackname" form found in
data/audio_features.csv; unknown songs are skipped with a warning.
`--playlist-file` expects a text file with one song per line.
"""

import argparse

import pandas as pd

from src import config
from src.data_prep import load_audio_features


def load_hindi_features():
    hindi_df = pd.read_csv(config.HINDI_FEATURES_PATH).drop_duplicates(subset='unique_track')
    return pd.concat([hindi_df['unique_track'],
                      hindi_df.drop(['id', 'unique_track'], axis=1)], axis=1)


def get_recommendations(method, playlist, library='english'):
    """Return a full (song, score) ranking of the target library, best first."""
    if method == 'knn':
        from src.knn import rank_songs
        target_df = load_hindi_features() if library == 'hindi' else None
        return rank_songs(playlist, target_df)

    if method == 'svd':
        if library == 'hindi':
            raise SystemExit("The SVD recommender only knows songs from the training "
                             "playlists — use --method knn or contrastive for --library hindi.")
        from src.svd import rank_songs
        return rank_songs(playlist)

    # contrastive (default)
    from src.contrastive import (load_song_features, load_trained_model,
                                 precompute_embeddings, recommend_songs)
    df_features = load_song_features()
    model = load_trained_model(df_features)
    embeddings = precompute_embeddings(model, df_features)
    if library == 'hindi':
        target = precompute_embeddings(model, load_hindi_features())
    else:
        target = embeddings
    return recommend_songs(embeddings, target, playlist)


def main():
    parser = argparse.ArgumentParser(
        description="Recommend songs for a playlist.",
        epilog='Song names use the "artistname - trackname" form from data/audio_features.csv.')
    parser.add_argument('--method', choices=['contrastive', 'knn', 'svd'],
                        default='contrastive', help='recommender to use (default: contrastive)')
    parser.add_argument('--songs', nargs='+', metavar='SONG',
                        help='playlist songs, e.g. "Coldplay - Fix You"')
    parser.add_argument('--playlist-file', metavar='FILE',
                        help='text file with one song per line (alternative to --songs)')
    parser.add_argument('--library', choices=['english', 'hindi'], default='english',
                        help='library to recommend from (default: english)')
    parser.add_argument('--top', type=int, default=20, metavar='N',
                        help='number of recommendations to show (default: 20)')
    args = parser.parse_args()

    if args.songs:
        playlist = args.songs
    elif args.playlist_file:
        with open(args.playlist_file, 'r', encoding='utf-8') as file:
            playlist = [line.strip() for line in file if line.strip()]
    else:
        parser.error('provide the playlist via --songs or --playlist-file')

    # Keep only songs the models know (they must be in the English library)
    known_songs = set(load_audio_features()['unique_track'])
    unknown = [song for song in playlist if song not in known_songs]
    for song in unknown:
        print(f"Skipping unknown song: {song}")
    playlist = [song for song in playlist if song in known_songs]
    if not playlist:
        raise SystemExit("None of the given songs are in the library "
                         "(see data/audio_features.csv for valid names).")

    recommendations = get_recommendations(args.method, playlist, args.library)

    score_name = 'distance' if args.method == 'knn' else 'similarity'
    print(f"\nTop {args.top} recommendations ({args.method}, {args.library} library) "
          f"for a playlist of {len(playlist)} songs:\n")
    for rank, (song, score) in enumerate(recommendations[:args.top], start=1):
        print(f"{rank:>3}. {song}  ({score_name} {score:.4f})")


if __name__ == '__main__':
    main()
