import streamlit as st
import torch
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- Load assets ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CF model
from ncf_model import NCF  # <- you’ll save your class into ncf_model.py
num_users, num_movies = 100000, 20000  # replace with actual values
model = NCF(num_users, num_movies)
model.load_state_dict(torch.load("ncf_model.pth", map_location=device))
model.eval().to(device)

# Load utils
user_map = joblib.load("user_map.pkl")
movie_map = joblib.load("movie_map.pkl")
movie_tfidf_idx = joblib.load("movie_tfidf_idx.pkl")
tfidf = joblib.load("tfidf.pkl")
movies = pd.read_csv("movies.csv")

# ---------------- Helper ----------------
def recommend_hybrid_from_titles(input_titles, top_n=10, cf_weight=0.7):
    # Find closest movies (case-insensitive + fuzzy)
    input_movies = []
    for title in input_titles:
        match = movies[movies["title"].str.lower().str.contains(title.lower(), na=False)]
        if not match.empty:
            input_movies.append(match.iloc[0]["movieId"])

    if not input_movies:
        return []

    # For each candidate movie, compute hybrid score
    candidates = movies["movieId"].tolist()
    results = []
    for m in candidates:
        # CF part
        movie_tensor = torch.tensor([movie_map.get(m, 0)], dtype=torch.long).to(device)
        user_tensor = torch.tensor([0], dtype=torch.long).to(device)  # dummy user
        cf_score = model(user_tensor, movie_tensor).item()

        # Content part
        sims = []
        for im in input_movies:
            if im in movie_tfidf_idx and m in movie_tfidf_idx:
                sims.append(
                    cosine_similarity(
                        tfidf.transform([movies.loc[movies.movieId==im, "genres"].values[0]]),
                        tfidf.transform([movies.loc[movies.movieId==m, "genres"].values[0]])
                    )[0,0]
                )
        content_score = np.mean(sims) if sims else 0
        final_score = cf_weight*cf_score + (1-cf_weight)*content_score
        results.append((m, final_score))

    recs = sorted(results, key=lambda x: -x[1])[:top_n]
    return movies[movies["movieId"].isin([r[0] for r in recs])][["title"]].values.flatten()

# ---------------- Streamlit UI ----------------
st.title("🎬 Hybrid Movie Recommender")
st.write("Enter movies you like, and get hybrid recommendations!")

input_text = st.text_area("Enter movies (comma separated):")
if input_text:
    titles = [t.strip() for t in input_text.split(",")]
    recs = recommend_hybrid_from_titles(titles, top_n=10)
    st.subheader("Recommended Movies:")
    for r in recs:
        st.write(r)
