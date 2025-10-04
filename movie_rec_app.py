import streamlit as st
import pandas as pd
import re
import torch
import pickle
import numpy as np
from rapidfuzz import process, fuzz
from ncf_model import NCF  # your model class

# -----------------------------
# Device setup
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------
# Load model, data, and mappings
# -----------------------------
with open("user_map.pkl", "rb") as f:
    user_map = pickle.load(f)
with open("movie_map.pkl", "rb") as f:
    movie_map = pickle.load(f)

num_users = len(user_map)
num_items = len(movie_map)

# Load NCF model
model = NCF(num_users, num_items).to(device)
model.load_state_dict(torch.load("ncf_model.pth", map_location=device))
model.eval()

# Load movies and content-based data
df = pd.read_csv("movies.csv")
with open("movie_tfidf_idx.pkl", "rb") as f:
    movie_tfidf_idx = pickle.load(f)

# -----------------------------
# Preprocessing
# -----------------------------
def normalize_title(title):
    return re.sub(r'\(\d{4}\)', '', title).strip().lower()

movie_title_map = {normalize_title(title): mid for mid, title in zip(df['movieId'], df['title'])}

def find_known_movies(user_input_list, movie_title_map, threshold=60):
    matched_ids = []
    for inp in user_input_list:
        inp = inp.lower().strip()
        best_match = process.extractOne(inp, movie_title_map.keys(), scorer=fuzz.token_sort_ratio)
        if best_match:
            match_title, score, _ = best_match
            if score >= threshold:
                matched_ids.append(movie_title_map[match_title])
    return matched_ids

# -----------------------------
# Content similarity helper
# -----------------------------
def content_similarity(mid1, mid2):
    vec1 = movie_tfidf_idx.get(mid1)
    vec2 = movie_tfidf_idx.get(mid2)
    if vec1 is None or vec2 is None:
        return 0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)

# -----------------------------
# Recommendation function
# -----------------------------
def recommend_from_movies(user_input_ids, model, top_n=10, top_k_ref=2, cf_weight=0.7, device='cpu', pop_penalty=True, candidate_k=100):
    """
    user_input_ids: list of movie IDs that the user likes
    model: pre-trained NCF model
    """
    model.eval()  # ensure model is in evaluation mode
    all_movies = df['movieId'].unique()
    
    # Candidate selection: movies not in user input
    candidates = [m for m in all_movies if m not in user_input_ids and m in movie_tfidf_idx]
    
    if not candidates:
        return []

    # Vectorized CF scoring
    user_tensor = torch.tensor([0]*len(candidates), dtype=torch.long).to(device)  # dummy user id 0 for inference
    movie_tensor = torch.tensor([movie_map[m] for m in candidates], dtype=torch.long).to(device)
    
    with torch.no_grad():
        cf_scores = model(user_tensor, movie_tensor).detach().cpu().numpy()  # detach before converting

    # Content-based scores
    content_scores = []
    for m in candidates:
        sims = [content_similarity(um, m) for um in user_input_ids if um in movie_tfidf_idx]
        content_scores.append(np.mean(sims) if sims else 0)
    content_scores = np.array(content_scores)

    # Popularity penalty
    if pop_penalty:
        popularity = df.groupby('movieId').size().to_dict()
        pop_scores = np.array([1 / (np.log(1 + popularity.get(m, 1))) for m in candidates])
    else:
        pop_scores = np.zeros(len(candidates))

    # Weighted hybrid score
    hybrid_scores = cf_weight * cf_scores + (1 - cf_weight) * content_scores + 0.1 * pop_scores

    # Top-N selection
    top_idx = np.argsort(-hybrid_scores)[:top_n]
    recommendations = []
    for i in top_idx:
        movie_id = candidates[i]
        h_score = round(hybrid_scores[i], 3)

        # Content-based explanation
        if len(user_input_ids) > 0:
            sims = [(m, content_similarity(m, movie_id)) for m in user_input_ids if m in movie_tfidf_idx]
            sims.sort(key=lambda x: x[1], reverse=True)
            top_refs = sims[:top_k_ref]
            ref_strings = [f"{df[df['movieId']==m]['title'].values[0]} (sim={sim:.2f})" for m, sim in top_refs]
            explanation = "Because you liked " + " and ".join(ref_strings)
        else:
            explanation = "Recommended based on your preferences"

        recommendations.append((df[df['movieId']==movie_id]['title'].values[0], explanation, h_score))

    return recommendations

# -----------------------------
# Streamlit App
# -----------------------------
st.title("🎬 Hybrid Movie Recommender")

user_input = st.text_area("Enter your favorite movies (comma-separated):", key="movie_input")

if st.button("Recommend"):
    if not user_input.strip():
        st.warning("Please enter at least one movie.")
    else:
        # Split input and match to known movies
        input_movies = [m.strip() for m in user_input.split(",")]
        known_movie_ids = find_known_movies(input_movies, movie_title_map)
        
        if not known_movie_ids:
            st.warning("No known movies found from your input. Check spelling or try another movie.")
        else:
            # ✅ Pass model and device to your recommendation function
            recommendations = recommend_from_movies(
                known_movie_ids,
                model=model,
                top_n=10,
                device=device
            )
            
            st.success(f"Found {len(recommendations)} recommendations!")
            for title, reason, score in recommendations:
                st.write(f"**{title}** — {reason} — Score: {score:.2f}")

