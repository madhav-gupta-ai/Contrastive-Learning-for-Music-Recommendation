"""Hybridized filtering via deep contrastive learning.

An MLP encoder maps each song's 8 audio features (content) into a 16-D
embedding. The embedding space geometry is learned with triplet loss on
playlist co-occurrence triplets (collaborative signal), hybridizing both
information sources. Recommendations: rank all songs by cosine similarity
of their embeddings to the mean embedding of the given playlist.

Outputs: results/model.pth, results/train_loss.csv
"""

import os

import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from src import config
from src.data_prep import load_audio_features


def load_song_features():
    """Song name + 8 feature columns, one row per song."""
    audio_features_df = load_audio_features()
    all_song_names = audio_features_df['unique_track']
    feature_values = audio_features_df.drop(['id', 'unique_track'], axis=1)
    return pd.concat([all_song_names, feature_values], axis=1)


class SongFeaturesDataset(Dataset):
    def __init__(self, triplets, df_features):
        self.df_features = df_features.set_index('unique_track')
        self.triplets = triplets
        self.feature_names = df_features.columns[1:]  # Excluding song_name

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_name, positive_name, negative_name = self.triplets[idx]
        anchor_features = self.df_features.loc[anchor_name, self.feature_names].values.astype('float32')
        positive_features = self.df_features.loc[positive_name, self.feature_names].values.astype('float32')
        negative_features = self.df_features.loc[negative_name, self.feature_names].values.astype('float32')
        return torch.tensor(anchor_features), torch.tensor(positive_features), torch.tensor(negative_features)


class ContrastiveNetwork(nn.Module):
    def __init__(self, input_size, layer_sizes):
        super(ContrastiveNetwork, self).__init__()
        layers = []
        for output_size in layer_sizes:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ReLU())
            input_size = output_size
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


def triplet_loss(anchor, positive, negative, margin=1.0):
    pos_dist = (anchor - positive).pow(2).sum(1)
    neg_dist = (anchor - negative).pow(2).sum(1)
    loss = torch.relu(pos_dist - neg_dist + margin)
    return loss.mean()


def train_model(model, dataloader, optimizer):
    """Train and return the mean triplet loss of each epoch."""
    epoch_losses = []
    for epoch in range(config.NUM_EPOCHS):
        losses = []
        for anchor, positive, negative in dataloader:
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)
            loss = triplet_loss(anchor_out, positive_out, negative_out, margin=config.MARGIN)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        epoch_losses.append(sum(losses) / len(losses))
        print(f'Epoch {epoch + 1}/{config.NUM_EPOCHS}, Loss: {epoch_losses[-1]}')
    return epoch_losses


def precompute_embeddings(model, df_features):
    model.eval()
    with torch.no_grad():
        # Prepare features for all songs
        features = torch.tensor(df_features.drop(columns='unique_track').values.astype('float32'))
        embeddings = model(features)

        # Store alongside unique_track identifiers
        embeddings_df = pd.DataFrame(embeddings.numpy())
        embeddings_df['unique_track'] = df_features['unique_track'].values

    return embeddings_df


def recommend_songs(playlist_embeddings_df, target_embeddings_df, playlist):
    # Calculate the average embedding for the playlist
    avg_playlist_embedding = playlist_embeddings_df[
        playlist_embeddings_df['unique_track'].isin(playlist)
    ].drop(columns='unique_track').mean().values.reshape(1, -1)

    # Retrieve embeddings for non-playlist songs from target_embeddings_df
    non_playlist_embeddings = target_embeddings_df[~target_embeddings_df['unique_track'].isin(playlist)]

    # Compute cosine similarity between the average playlist embedding and each
    # non-playlist song embedding; recommend the most similar first
    similarities = cosine_similarity(avg_playlist_embedding,
                                     non_playlist_embeddings.drop(columns='unique_track').values)[0]

    recommendations = list(zip(non_playlist_embeddings['unique_track'], similarities))
    recommendations.sort(key=lambda x: x[1], reverse=True)

    return recommendations


def load_trained_model(df_features):
    """Load the trained encoder from results/model.pth."""
    input_size = len(df_features.columns) - 1  # one column is 'unique_track'
    model = ContrastiveNetwork(input_size, config.LAYER_SIZES)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.eval()
    return model


def main():
    torch.manual_seed(config.RANDOM_SEED)

    df_features = load_song_features()
    triplets = list(pd.read_csv(config.TRIPLETS_PATH).to_records(index=False))
    print(f"Training on {len(triplets)} triplets")

    # Dataset and DataLoader
    dataset = SongFeaturesDataset(triplets, df_features)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # Model and optimizer
    input_size = len(df_features.columns) - 1  # one column is 'unique_track'
    model = ContrastiveNetwork(input_size, config.LAYER_SIZES)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    epoch_losses = train_model(model, dataloader, optimizer)

    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_PATH)
    pd.DataFrame({'epoch': range(1, len(epoch_losses) + 1),
                  'mean_triplet_loss': epoch_losses}).to_csv(config.TRAIN_LOSS_PATH, index=False)
    print(f"Saved: {config.MODEL_PATH}, {config.TRAIN_LOSS_PATH}")


if __name__ == '__main__':
    main()
