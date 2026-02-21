"""
Neural Collaborative Filtering (NCF) Recommender
=================================================
A PyTorch-based neural network that learns user and item embeddings
to predict ratings. Combines matrix factorization with an MLP so it
can capture both linear and non-linear user-item interactions.

Architecture
------------
    User ID ──► User Embedding ──┐
                                  ├──► Concat ──► MLP ──► Sigmoid ──► Predicted Rating
    Item ID ──► Item Embedding ──┘

Training
--------
    python neural_recommender.py

    Expects data/ratings.csv (MyAnimeList or MovieLens format).
    See README for download links. Saves model to models/ncf_model.pt
"""

import os
import time
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

class Config:
    # Paths — separate files per dataset so they never overwrite each other
    ANIME_RATINGS_PATH  = "data/anime_ratings.csv"   # MyAnimeList: rename rating.csv → anime_ratings.csv
    MOVIE_RATINGS_PATH  = "data/movie_ratings.csv"   # MovieLens:   rename ratings.csv → movie_ratings.csv
    ANIME_PATH          = "data/anime.csv"
    MODEL_DIR           = "models"
    MODEL_PATH          = "models/ncf_model.pt"
    ENCODER_PATH        = "models/encoders.pkl"

    # Model hyperparameters
    EMBEDDING_DIM  = 64        # size of user/item embedding vectors
    MLP_LAYERS     = [128, 64, 32]  # hidden layer sizes
    DROPOUT        = 0.3

    # Training hyperparameters
    EPOCHS         = 20
    BATCH_SIZE     = 1024
    LEARNING_RATE  = 1e-3
    WEIGHT_DECAY   = 1e-5
    VAL_SPLIT      = 0.1
    MIN_RATINGS    = 5         # drop users/items with fewer than this many ratings
    MAX_RATINGS    = 10_000_000  # cap dataset size to keep training manageable

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class RatingsDataset(Dataset):
    """PyTorch Dataset wrapping (user_id, item_id, rating) triples."""

    def __init__(self, users, items, ratings):
        self.users   = torch.tensor(users,   dtype=torch.long)
        self.items   = torch.tensor(items,   dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────

class NCF(nn.Module):
    """
    Neural Collaborative Filtering model.

    Combines:
    - GMF path: element-wise product of embeddings (linear interactions)
    - MLP path: concatenated embeddings through fully-connected layers (non-linear)
    Both paths are merged and passed through a final output layer.
    """

    def __init__(self, n_users: int, n_items: int, cfg: Config):
        super().__init__()

        # Embeddings — one set for GMF, one for MLP
        self.user_emb_gmf = nn.Embedding(n_users, cfg.EMBEDDING_DIM)
        self.item_emb_gmf = nn.Embedding(n_items, cfg.EMBEDDING_DIM)

        self.user_emb_mlp = nn.Embedding(n_users, cfg.EMBEDDING_DIM)
        self.item_emb_mlp = nn.Embedding(n_items, cfg.EMBEDDING_DIM)

        # MLP tower
        mlp_input_dim = cfg.EMBEDDING_DIM * 2
        layers = []
        in_dim = mlp_input_dim
        for out_dim in cfg.MLP_LAYERS:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(cfg.DROPOUT),
            ]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        # Final output — merges GMF + MLP
        self.output = nn.Linear(cfg.EMBEDDING_DIM + cfg.MLP_LAYERS[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for emb in [self.user_emb_gmf, self.item_emb_gmf,
                    self.user_emb_mlp, self.item_emb_mlp]:
            nn.init.normal_(emb.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, user_ids, item_ids):
        # GMF path
        u_gmf = self.user_emb_gmf(user_ids)
        i_gmf = self.item_emb_gmf(item_ids)
        gmf_out = u_gmf * i_gmf  # element-wise product

        # MLP path
        u_mlp = self.user_emb_mlp(user_ids)
        i_mlp = self.item_emb_mlp(item_ids)
        mlp_in = torch.cat([u_mlp, i_mlp], dim=1)
        mlp_out = self.mlp(mlp_in)

        # Merge and predict
        merged = torch.cat([gmf_out, mlp_out], dim=1)
        out = torch.sigmoid(self.output(merged))  # → (0, 1)
        return out.squeeze()


# ──────────────────────────────────────────────
# Data Loading & Preprocessing
# ──────────────────────────────────────────────

def _load_single(path: str, source_tag: str) -> pd.DataFrame:
    """Load one ratings CSV and normalise its columns."""
    print(f"  Loading {source_tag} from {path} ...")
    df = pd.read_csv(path)

    col_map = {"userId": "user_id", "movieId": "item_id", "anime_id": "item_id"}
    df = df.rename(columns=col_map)

    if "item_id" not in df.columns:
        for c in df.columns:
            if "id" in c.lower() and c != "user_id":
                df = df.rename(columns={c: "item_id"})
                break

    df = df[["user_id", "item_id", "rating"]].dropna()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df[df["rating"] > 0].dropna()

    # Prefix user IDs with source tag so anime users and movie users
    # don't accidentally share the same ID space
    df["user_id"] = source_tag + "_" + df["user_id"].astype(str)
    df["item_id"] = source_tag + "_" + df["item_id"].astype(str)
    return df


def load_ratings(cfg: Config):
    """
    Load and merge all available ratings files.
    Expects at least one of:
      data/anime_ratings.csv  (MyAnimeList — rename rating.csv to this)
      data/movie_ratings.csv  (MovieLens   — rename ratings.csv to this)
    """
    frames = []

    if os.path.exists(cfg.ANIME_RATINGS_PATH):
        frames.append(_load_single(cfg.ANIME_RATINGS_PATH, "anime"))
    else:
        print(f"  [skip] {cfg.ANIME_RATINGS_PATH} not found")

    if os.path.exists(cfg.MOVIE_RATINGS_PATH):
        frames.append(_load_single(cfg.MOVIE_RATINGS_PATH, "movie"))
    else:
        print(f"  [skip] {cfg.MOVIE_RATINGS_PATH} not found")

    if not frames:
        raise FileNotFoundError(
            "No ratings files found. Download the datasets and rename them:\n"
            "  MyAnimeList rating.csv  → data/anime_ratings.csv\n"
            "  MovieLens   ratings.csv → data/movie_ratings.csv"
        )

    df = pd.concat(frames, ignore_index=True)

    # Remove users/items with very few ratings
    user_counts = df["user_id"].value_counts()
    item_counts = df["item_id"].value_counts()
    df = df[
        df["user_id"].isin(user_counts[user_counts >= cfg.MIN_RATINGS].index) &
        df["item_id"].isin(item_counts[item_counts >= cfg.MIN_RATINGS].index)
    ]

    print(f"  Total before sampling: {len(df):,} ratings")

    # Sample down if dataset is huge — keeps training fast without losing much quality
    if len(df) > cfg.MAX_RATINGS:
        df = df.sample(cfg.MAX_RATINGS, random_state=42)
        print(f"  Sampled down to {cfg.MAX_RATINGS:,} ratings")

    print(f"  Final: {len(df):,} ratings | {df['user_id'].nunique():,} users | {df['item_id'].nunique():,} items")
    return df


def encode_and_normalise(df: pd.DataFrame):
    """
    Label-encode user/item IDs to contiguous integers.
    Normalise ratings to [0, 1].
    Returns encoded df + the encoders (needed at inference time).
    """
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()

    df = df.copy()
    df["user_idx"] = user_enc.fit_transform(df["user_id"])
    df["item_idx"] = item_enc.fit_transform(df["item_id"])

    # Normalise rating to [0, 1]
    r_min, r_max = df["rating"].min(), df["rating"].max()
    df["rating_norm"] = (df["rating"] - r_min) / (r_max - r_min)

    encoders = {
        "user_enc": user_enc,
        "item_enc": item_enc,
        "rating_min": r_min,
        "rating_max": r_max,
        "n_users": df["user_idx"].nunique(),
        "n_items": df["item_idx"].nunique(),
    }
    return df, encoders


# ──────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────

def train(cfg: Config = None):
    cfg = cfg or Config()
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    print(f"Device: {cfg.DEVICE}\n")

    # ── Load data ──
    df = load_ratings(cfg)
    df, encoders = encode_and_normalise(df)

    train_df, val_df = train_test_split(df, test_size=cfg.VAL_SPLIT, random_state=42)

    train_ds = RatingsDataset(
        train_df["user_idx"].values,
        train_df["item_idx"].values,
        train_df["rating_norm"].values,
    )
    val_ds = RatingsDataset(
        val_df["user_idx"].values,
        val_df["item_idx"].values,
        val_df["rating_norm"].values,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Build model ──
    model = NCF(encoders["n_users"], encoders["n_items"], cfg).to(cfg.DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.LEARNING_RATE,
                                 weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    history = {"train_loss": [], "val_loss": [], "val_rmse": []}
    best_val_loss = float("inf")

    # ── Training loop ──
    for epoch in range(1, cfg.EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for users, items, ratings in train_loader:
            users, items, ratings = (
                users.to(cfg.DEVICE),
                items.to(cfg.DEVICE),
                ratings.to(cfg.DEVICE),
            )
            optimizer.zero_grad()
            preds = model(users, items)
            loss = criterion(preds, ratings)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(ratings)

        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for users, items, ratings in val_loader:
                users, items, ratings = (
                    users.to(cfg.DEVICE),
                    items.to(cfg.DEVICE),
                    ratings.to(cfg.DEVICE),
                )
                preds = model(users, items)
                val_loss += criterion(preds, ratings).item() * len(ratings)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(ratings.cpu().numpy())

        val_loss /= len(val_ds)

        # RMSE in original rating scale
        r_range = encoders["rating_max"] - encoders["rating_min"]
        rmse = np.sqrt(np.mean((np.array(all_preds) - np.array(all_targets)) ** 2)) * r_range

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(rmse)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:>3}/{cfg.EPOCHS}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_rmse={rmse:.3f}  "
            f"({elapsed:.1f}s)"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "config": cfg,
                "encoders": encoders,
                "history": history,
            }, cfg.MODEL_PATH)
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    print(f"Model saved to {cfg.MODEL_PATH}")

    # Also save encoders separately for easy access
    with open(cfg.ENCODER_PATH, "wb") as f:
        pickle.dump(encoders, f)

    return model, encoders, history


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────

class NeuralRecommender:
    """
    Wraps the trained NCF model for inference.

    Usage
    -----
    rec = NeuralRecommender.load()
    recs = rec.recommend_for_user(user_id=42, anime_df=df, n=10)
    recs = rec.recommend_from_ratings({"Attack on Titan": 9, "Arcane": 8}, anime_df=df)
    """

    def __init__(self, model: NCF, encoders: dict, cfg: Config):
        self.model    = model
        self.encoders = encoders
        self.cfg      = cfg
        self.model.eval()

    @classmethod
    def load(cls, path: str = "models/ncf_model.pt"):
        checkpoint = torch.load(path, map_location="cpu")
        cfg        = checkpoint["config"]
        encoders   = checkpoint["encoders"]
        model      = NCF(encoders["n_users"], encoders["n_items"], cfg)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded model from epoch {checkpoint['epoch']} (val_loss={min(checkpoint['history']['val_loss']):.4f})")
        return cls(model, encoders, cfg)

    def recommend_for_user(
        self,
        user_id: int,
        item_df: pd.DataFrame,
        watched_item_ids: list = None,
        n: int = 10,
    ) -> pd.DataFrame:
        """
        Generate top-N recommendations for a known user in the training set.
        item_df should have columns: item_id, title, type, genres, score
        """
        enc = self.encoders
        if user_id not in enc["user_enc"].classes_:
            raise ValueError(f"User {user_id} not in training data. Use recommend_from_ratings() instead.")

        user_idx = enc["user_enc"].transform([user_id])[0]

        # Only score items we know about
        known_items = item_df[item_df["item_id"].isin(enc["item_enc"].classes_)].copy()
        if watched_item_ids:
            known_items = known_items[~known_items["item_id"].isin(watched_item_ids)]

        item_idxs = enc["item_enc"].transform(known_items["item_id"])
        user_idxs = np.full(len(item_idxs), user_idx)

        with torch.no_grad():
            preds = self.model(
                torch.tensor(user_idxs, dtype=torch.long),
                torch.tensor(item_idxs, dtype=torch.long),
            ).numpy()

        # Scale back to original rating range
        r_min, r_max = enc["rating_min"], enc["rating_max"]
        preds_scaled = preds * (r_max - r_min) + r_min

        known_items = known_items.copy()
        known_items["predicted_rating"] = preds_scaled
        return known_items.nlargest(n, "predicted_rating")[
            ["title", "type", "genres", "score", "predicted_rating"]
        ].reset_index(drop=True)

    def recommend_from_ratings(
        self,
        title_ratings: dict,        # {"Attack on Titan": 9, "Arcane": 8}
        item_df: pd.DataFrame,      # the full anime/movie DataFrame
        n: int = 10,
        filter_type: str = None,
    ) -> pd.DataFrame:
        """
        For new/cold-start users: find the most similar known user
        based on item overlap, then use their embedding for predictions.
        Falls back to content-based if no overlap found.
        """
        enc = self.encoders

        # Map title → item_id
        title_to_id = dict(zip(item_df["title"].str.lower(), item_df["item_id"]))
        rated_ids = {}
        for title, rating in title_ratings.items():
            iid = title_to_id.get(title.lower())
            if iid is not None and iid in enc["item_enc"].classes_:
                rated_ids[iid] = rating

        if not rated_ids:
            print("No rated items found in model's known items. Try content-based fallback.")
            return pd.DataFrame()

        # Build a pseudo-user embedding by averaging the item embeddings
        # weighted by the user's ratings
        item_idxs = enc["item_enc"].transform(list(rated_ids.keys()))
        weights   = torch.tensor(
            [(r / 10.0) for r in rated_ids.values()], dtype=torch.float32
        ).unsqueeze(1)

        with torch.no_grad():
            item_embs = self.model.item_emb_mlp(torch.tensor(item_idxs, dtype=torch.long))
            pseudo_user_emb = (item_embs * weights).sum(dim=0) / weights.sum()

            # Score all items
            candidates = item_df[
                item_df["item_id"].isin(enc["item_enc"].classes_) &
                ~item_df["item_id"].isin(rated_ids.keys())
            ].copy()

            if filter_type:
                candidates = candidates[candidates["type"] == filter_type]

            all_item_idxs = enc["item_enc"].transform(candidates["item_id"])
            all_item_embs = self.model.item_emb_mlp(
                torch.tensor(all_item_idxs, dtype=torch.long)
            )
            # Cosine similarity between pseudo user and each item
            scores = torch.nn.functional.cosine_similarity(
                pseudo_user_emb.unsqueeze(0).expand_as(all_item_embs),
                all_item_embs
            ).numpy()

        candidates["neural_score"] = scores
        results = candidates.nlargest(n, "neural_score")[
            ["title", "type", "genres", "score", "neural_score"]
        ].reset_index(drop=True)
        return results


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    cfg = Config()

    anime_ok = os.path.exists(cfg.ANIME_RATINGS_PATH)
    movie_ok = os.path.exists(cfg.MOVIE_RATINGS_PATH)

    if not anime_ok and not movie_ok:
        print("ERROR: No ratings files found.")
        print("Download and rename:")
        print("  MyAnimeList rating.csv  → data/anime_ratings.csv")
        print("  https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database")
        print()
        print("  MovieLens ratings.csv   → data/movie_ratings.csv")
        print("  https://grouplens.org/datasets/movielens/")
    else:
        train(cfg)
