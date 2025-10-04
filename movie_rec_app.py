import streamlit as st
import pandas as pd
import re
import torch
import pickle
import joblib
from rapidfuzz import process, fuzz
from ncf_model import NCF

# -----------------------------
# Device
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------
# Load user/movie maps
# -----------------------------
with open("user_map.pkl", "rb") as f:
    user_map = pickle.load(f)
with open("movie_map.pkl", "rb") as f:
    movie_map = pickle.load(f)

# -----------------------------
# Load TF-IDF index safely
# -----------------------------
movie_tfidf_idx = joblib.load("tfidf.pkl")  # use joblib for large sparse objects

# -----------------------------
# Load movies data
# -----------------------------
df = pd.read_csv("movies.csv")

# -----------------------------
# Load NCF model
# -----------------------------
num_users = len(user_map)
num_items = len(movie_map)

model = NCF(num_users, num_items).to(device)
model.load_state_dict(torch.load("ncf_model.pth", map_location=device))
model.eval()

# -----------------------------
# Preprocessing
# -----------------------------
def normalize_title(title):
    return re.sub(r'\(\d{4}\)', '', title).strip().lower()

movie_title_map = {normalize_title(title): mid for mid, title in zip(df['movieId'], df['title'])}

# -----------------------------
# Fuzzy match input
# -----------------------------
def find_known_movies(user_input_list, movie_title_map, threshold=60):
    matched_ids = []
    for inp in user_input_list:
        inp = inp.lower().strip()
        # extractOne now returns (match, score, index)
        best_match, score, _ = process.extractOne(
            inp, movie_title_map.keys(), scorer=fuzz.token_sort_ratio
        )
        if score >= threshold:
            matched_ids.append(movie_title_map[best_match])
    return matched_ids

# -----------------------------
# Hybrid recommendation function
# -----------------------------
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def content_similarity(m1, m2):
    """Compute cosine similarity between two movies in TF-IDF space"""
    return cosine_similarity(movie_tfidf_idx[m1], movie_tfidf_idx[m2])[0,0]

def recommend_hybrid_from_list(input_movie_ids, top_n=10, top_k_ref=2, cf_weight=0.7):
    all_movies = df['movieId'].values
    candidates = [m for m in all_movies if m not in input_movie_ids]
    
    if not candidates:
        return []

    # CF scoring
    user_tensor = torch.tensor([0]*len(candidates), dtype=torch.long).to(device)  # dummy user
    movie_tensor = torch.tensor([movie_map[m] for m in candidates], dtype=torch.long).to(device)
    with torch.no_grad():
        cf_scores = model(user_tensor, movie_tensor).cpu().numpy()

    # Content-based scoring
    content_scores = []
    for m in candidates:
        sims = [content_similarity(im, m) for im in input_movie_ids if im in movie_tfidf_idx]
        content_scores.append(np.mean(sims) if sims else 0)
    content_scores = np.array(content_scores)

    # Weighted hybrid
    hybrid_scores = cf_weight * cf_scores + (1 - cf_weight) * content_scores

    # Top-N
    top_idx = np.argsort(-hybrid_scores)[:top_n]
    recommendations = []
    for i in top_idx:
        movie_id = candidates[i]
        score = hybrid_scores[i]
        explanation = "Recommended based on your input movies"
        recommendations.append((df[df['movieId']==movie_id]['title'].values[0], explanation, score))

    return recommendations

# -----------------------------
# Streamlit app
# -----------------------------
st.title("🎬 Hybrid Movie Recommender")
user_input = st.text_area("Enter your favorite movies (comma-separated):", key="movies_input")

if st.button("Recommend"):
    if not user_input.strip():
        st.warning("Please enter at least one movie.")
    else:
        input_movies = [m.strip() for m in user_input.split(",")]
        known_movie_ids = find_known_movies(input_movies, movie_title_map)
        
        if not known_movie_ids:
            st.warning("No known movies found from your input. Check spelling or try another movie.")
        else:
            recommendations = recommend_hybrid_from_list(known_movie_ids, top_n=10)
            st.success(f"Found {len(recommendations)} recommendations!")
            for title, reason, score in recommendations:
                st.write(f"**{title}** — {reason} — Score: {score:.2f}")
