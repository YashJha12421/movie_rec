import streamlit as st
import pandas as pd
import re
import pickle
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz
from ncf_model import NCF
import joblib

# -----------------------------
# Device setup
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------
# Load data and mappings
# -----------------------------
df = pd.read_csv("movies.csv")

with open("user_map.pkl", "rb") as f:
    user_map = pickle.load(f)
with open("movie_map.pkl", "rb") as f:
    movie_map = pickle.load(f)

num_users = len(user_map)
num_items = len(movie_map)

# -----------------------------
# Load TF-IDF vectorizer & transform movie titles
# -----------------------------
vectorizer = joblib.load("tfidf.pkl")
movie_tfidf_idx = vectorizer.transform(df['title'])
movieId_to_idx = {mid: idx for idx, mid in enumerate(df['movieId'].values)}

# -----------------------------
# Load NCF model
# -----------------------------
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
# Fuzzy matching
# -----------------------------
def find_known_movies(user_input_list, movie_title_map, threshold=60):
    matched_ids = []
    for inp in user_input_list:
        inp = inp.lower().strip()
        # extractOne returns (match, score, index)
        best_match, score, _ = process.extractOne(inp, movie_title_map.keys(), scorer=fuzz.token_sort_ratio)
        if score >= threshold:
            matched_ids.append(movie_title_map[best_match])
    return matched_ids

# -----------------------------
# Content similarity
# -----------------------------
def content_similarity(m1, m2):
    idx1 = movieId_to_idx[m1]
    idx2 = movieId_to_idx[m2]
    return cosine_similarity(movie_tfidf_idx[idx1], movie_tfidf_idx[idx2])[0,0]

# -----------------------------
# Hybrid recommendation with diversity
# -----------------------------
def recommend_hybrid_from_list(input_movie_ids, top_n=10, top_k_ref=2, model=model, cf_weight=0.7, device=device, pop_penalty=True, candidate_k=50):
    model.eval()
    # Collect candidate movies
    all_movie_ids = df['movieId'].values
    candidates = set()
    for m in input_movie_ids:
        sims = [(mid, content_similarity(m, mid)) for mid in all_movie_ids if mid not in input_movie_ids]
        sims.sort(key=lambda x: x[1], reverse=True)
        candidates.update([mid for mid, _ in sims[:candidate_k]])
    candidates = list(candidates)
    if not candidates:
        return []

    # Collaborative filtering scores
    user_tensor = torch.tensor([0]*len(candidates), dtype=torch.long).to(device)  # dummy user 0 since we don't have real user
    movie_tensor = torch.tensor([movie_map[m] for m in candidates], dtype=torch.long).to(device)
    with torch.no_grad():
        cf_scores = model(user_tensor, movie_tensor).cpu().numpy()

    # Content scores
    content_scores = []
    for m in candidates:
        sims = [content_similarity(im, m) for im in input_movie_ids if im in movieId_to_idx]
        content_scores.append(np.mean(sims) if sims else 0)
    content_scores = np.array(content_scores)

    # Popularity penalty
    if pop_penalty:
        popularity = df.groupby('movieId').size().to_dict()
        pop_scores = np.array([1 / (np.log(1 + popularity.get(m,1))) for m in candidates])
    else:
        pop_scores = np.zeros(len(candidates))

    # Hybrid score
    hybrid_scores = cf_weight * cf_scores + (1 - cf_weight) * content_scores + 0.1 * pop_scores

    # Top-N selection
    top_idx = np.argsort(-hybrid_scores)[:top_n]
    recommendations = []
    for i in top_idx:
        movie_id = candidates[i]
        h_score = round(hybrid_scores[i], 3)
        explanation = "Recommended based on your input movies"
        recommendations.append((df[df['movieId']==movie_id]['title'].values[0], explanation, h_score))

    return recommendations

# -----------------------------
# Streamlit app
# -----------------------------
st.title("🎬 Hybrid Movie Recommender")

user_input = st.text_area("Enter your favorite movies (comma-separated):", key="unique_text_area")

if st.button("Recommend"):
    if not user_input.strip():
        st.warning("Please enter at least one movie.")
    else:
        input_movies = [m.strip() for m in user_input.split(",")]
        known_movie_ids = find_known_movies(input_movies, movie_title_map)
        
        if not known_movie_ids:
            st.warning("No known movies found from your input. Check spelling or try another movie.")
        else:
            recommendations = recommend_hybrid_from_list(known_movie_ids)
            st.success(f"Found {len(recommendations)} recommendations!")
            for title, reason, score in recommendations:
                st.write(f"**{title}** — {reason} — Score: {score:.2f}")
