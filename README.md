# Global Music Recommendation using Deep Contrastive Learning and Hybridized Filtering

Code for the paper:

> Madhav Gupta and Jayaraman K. Valadi, **"Global Music Recommendation using
> Deep Contrastive Learning and Hybridized Filtering,"** *2025 International
> Conference on Innovation in Computing and Engineering (ICE)*, Greater Noida,
> India, Feb 2025, pp. 1–5.
> DOI: [10.1109/ICE63309.2025.10984218](https://doi.org/10.1109/ICE63309.2025.10984218)

A music recommender trained on ~13,000 real Spotify playlists. A neural
encoder learns — via triplet loss on playlist co-occurrence — to map a song's
8 Spotify audio features into an embedding space where songs that listeners
enjoy together sit close, blending content and collaborative signals in a
single model. Since any song with audio features can be embedded, the model
also recommends across languages: the included demo suggests Hindi songs for
an English playlist. Classic content-based (KNN) and collaborative-filtering
(SVD) recommenders are included alongside it.

## Setup

Python 3.10/3.11:

```
pip install -r requirements.txt
```

The trained models ship with the repository (`results/`), so recommendations
work out of the box. To retrain from scratch, download the raw dataset (see
[data/README.md](data/README.md)) and run, in order:

```
python -m src.data_prep
python -m src.create_triplets
python -m src.contrastive
python -m src.svd
```

## Usage

```
python -m src.recommend --songs "Coldplay - Fix You" "Adele - Someone Like You"
```

| Option | Meaning |
|---|---|
| `--method contrastive\|knn\|svd` | recommender to use (default: `contrastive`) |
| `--library english\|hindi` | library to recommend from (default: `english`) |
| `--top N` | number of recommendations (default: 20) |
| `--playlist-file FILE` | one song per line, instead of `--songs` |

Song names use the `"artist - title"` form found in `data/audio_features.csv`;
songs outside that library are skipped with a warning. `--playlist-file`
expects a plain-text file (UTF-8) with one song per line:

```
# my_playlist.txt
Coldplay - Fix You
Adele - Someone Like You
Bon Iver - Skinny Love
```

```
python -m src.recommend --playlist-file my_playlist.txt
```

## Notebooks

- [training.ipynb](notebooks/training.ipynb) — training curve and example recommendations
- [global_recommendations_demo.ipynb](notebooks/global_recommendations_demo.ipynb) — Hindi songs for an English playlist

## License

The code is released under the [MIT License](LICENSE) © 2024 Madhav Gupta. The Spotify Playlists
dataset and the Spotify audio features remain the property of their
respective owners.
