# Data

## Raw dataset (download required)

`spotify_dataset.csv` (~1.2 GB, not committed) — the **"Spotify Playlists"**
dataset: a crawl of #nowplaying tweets published via Spotify (Pichl, Zangerle
& Specht, ICDM Workshops 2015).

Download from Kaggle and place it in this folder:
https://www.kaggle.com/datasets/andrewmvd/spotify-playlists

Schema: `user_id` (SHA-hashed), `artistname`, `trackname`, `playlistname` —
comma-separated, `"`-quoted, `\`-escaped.

## Committed files

| File | Contents |
|---|---|
| `audio_features.csv` | 8 standardized (z-scored) Spotify audio features per song, plus Spotify `id` and `unique_track` ("artist - title"). Used by the KNN and contrastive models. |
| `audio_features_raw.csv` | The raw Spotify API payload the above was derived from. Both files are committed because Spotify deprecated the `audio-features` endpoint for new API apps in November 2024, so they can no longer be re-fetched (`src/fetch_audio_features.py` is kept for provenance). |
| `track_csv.csv` | Spotify track id ↔ song name mapping from the API search pass. |
| `hindi_features.csv` | Audio features for 1,801 popular Hindi songs (standardized with the same scaler), used by the global-recommendations demo. |

## Generated files (not committed)

Produced by the pipeline scripts, in order:

| File | Produced by |
|---|---|
| `full_dataset.csv` | `src/data_prep.py` — cleaned, re-keyed, unfiltered checkpoint |
| `filtered_dataset.csv` | `src/data_prep.py` — the working dataset after iterative filtering |
| `triplets.csv` | `src/create_triplets.py` — (anchor, positive, negative) training triplets |
