# 🎬 Media Recommendation System
**Movies · TV Shows · Anime**

A content-based recommendation engine that suggests what to watch next based on your viewing history. Built with Python, scikit-learn, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)

---

## 🚀 Demo

Open `demo.ipynb` to see the full walkthrough — no datasets required, runs on a built-in sample of 30 titles.

## ⚙️ How It Works

1. Each title's **genres + overview** are combined into a text "soup"
2. **TF-IDF** converts that text into a numeric vector
3. **Cosine similarity** measures how close any two titles are
4. Given your watched list (+ optional ratings), the engine scores every unseen title and returns the top matches

## 📁 Project Structure

```
media-recommender/
├── recommender.py     # Core recommendation engine
├── app.py             # Streamlit web app
├── demo.ipynb         # Jupyter notebook walkthrough
├── requirements.txt
├── data/              # Put your CSV datasets here
│   ├── movies.csv
│   ├── shows.csv
│   └── anime.csv
└── README.md
```

## 📥 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download datasets
Place these CSVs in the `data/` folder:

| File | Source |
|------|--------|
| `data/movies.csv` | [TMDB Movies – Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) |
| `data/shows.csv` | [TMDB TV Shows – Kaggle](https://www.kaggle.com/datasets/asaniczka/full-tmdb-tv-shows-dataset-2023) |
| `data/anime.csv` | [MyAnimeList – Kaggle](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database) |

### 3. Run the notebook demo
```bash
jupyter notebook demo.ipynb
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

## 🧪 Use the Recommender in Python

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

## 🔮 Possible Improvements

- **Collaborative filtering** with the `Surprise` library for user-based recommendations
- **Sentence-BERT embeddings** for richer semantic similarity
- **TMDB API** integration for live posters and metadata
- **User feedback loop** — mark results as "not interested" to refine future recommendations

## 📄 License

MIT
