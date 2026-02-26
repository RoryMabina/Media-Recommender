# Media Recommendation System
**Movies · TV Shows · Anime**

A hybrid recommendation engine that combines content-based filtering and a trained Neural Collaborative Filtering (NCF) model to suggest what to watch next. Built with Python, PyTorch, scikit-learn, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)

---

## Demo

Open `Media Recommendation System (Demo Notebook).ipynb` for a full walkthrough — no datasets required, runs on a built-in sample of 30 titles and synthetic rating data.

---

## How It Works

This project uses two recommendation approaches combined into a hybrid system:

**Content-Based Filtering (`recommender.py`)**
- Each title's genres and overview are combined into a weighted text representation
- TF-IDF converts that text into a numeric vector (genres weighted 3x higher than overview)
- Cosine similarity measures how close any two titles are
- Works immediately with no training data — just your watched list

**Neural Collaborative Filtering (`neural_recommender.py`)**
- A PyTorch NCF model with two paths: GMF (linear interactions) and MLP (non-linear patterns)
- Each user and item gets a learned 64-dimensional embedding vector
- Trained on 10M+ ratings from MyAnimeList and MovieLens combined
- For cold-start users, builds a pseudo-embedding by averaging rated item embeddings

---

## Project Structure

```
media-recommender/
├── recommender.py          # Content-based engine (TF-IDF + cosine similarity)
├── neural_recommender.py   # NCF model definition, training loop, inference
├── app.py                  # Streamlit web app
├── Media Recommendation System (Demo Notebook).ipynb              # Jupyter notebook walkthrough
├── requirements.txt
├── data/                   # Datasets go here (excluded from git)
│   ├── movies.csv
│   ├── anime.csv
│   ├── anime_ratings.csv
│   └── movie_ratings.csv
├── models/                 # Saved model weights (excluded from git)
│   └── ncf_model.pt
└── README.md
```

---

## Getting Started

### 1. Clone and install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/media-recommender.git
cd media-recommender
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 2. Download datasets

Place these CSVs in the `data/` folder:

| File | Source | Notes |
|------|--------|-------|
| `data/movies.csv` | [TMDB Movies – Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) | Rename to `movies.csv` |
| `data/anime.csv` | [MyAnimeList – Kaggle](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database) | Keep as `anime.csv` |
| `data/anime_ratings.csv` | Same Kaggle link above | Rename `rating.csv` to this |
| `data/movie_ratings.csv` | [MovieLens – GroupLens](https://grouplens.org/datasets/movielens/) | Rename `ratings.csv` to this |

### 3. Run the notebook demo

```bash
jupyter notebook demo.ipynb
```

### 4. Train the neural model (optional but recommended)

```bash
python neural_recommender.py
```

Trains for 20 epochs on up to 10M ratings. Takes around 60-90 minutes on CPU, 10-20 minutes with a CUDA GPU. The best checkpoint is saved automatically to `models/ncf_model.pt`.

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

---

## Use the Recommender in Python

```python
from recommender import load_data, MediaRecommender

df = load_data()
rec = MediaRecommender()
rec.fit(df)

recs = rec.recommend(
    watched=["Attack on Titan", "Breaking Bad", "Inception"],
    ratings={"Attack on Titan": 10, "Breaking Bad": 9, "Inception": 9},
    n=10,
)
print(recs)
```

---

## Model Architecture

```
User ID --> User Embedding (64d) --+
                                    +--(GMF)--> element-wise product --+
Item ID --> Item Embedding (64d) --+                                   +--> Linear --> Rating
                                    +--(MLP)--> [128 -> 64 -> 32] ----+
User ID --> User Embedding (64d) --+
                                    |
Item ID --> Item Embedding (64d) --+
```

Total parameters: ~40M. Trained with Adam optimiser, MSE loss, and learning rate scheduling on plateau.

---

## Ideas to Take It Further

- TMDB API integration for live posters and metadata
- Sentence-BERT embeddings to replace TF-IDF for richer semantic similarity
- User accounts with a database to persist watch history between sessions
- Feedback loop — mark results as "not interested" to refine future recommendations
- Sequence-aware model using a Transformer to capture the order titles were watched

---

## License

MIT
