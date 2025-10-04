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
def recommend_from_movies(input_movie_ids, model, top_n=10, top_k_ref=2, cf_weight=0.7, device='cpu', pop_penalty=True, candidate_k=50):
    model.eval()
    all_movies = set(df['movieId'].unique())
    rated_set = set(input_movie_ids)
    candidates = set()

    # 1️⃣ Candidate selection
    for m in input_movie_ids:
        if m not in movie_tfidf_idx:
            continue  # skip if no embeddings
        sims = [(mid, content_similarity(m, mid)) for mid in all_movies - rated_set if mid in movie_tfidf_idx]
        sims.sort(key=lambda x: x[1], reverse=True)
        candidates.update([mid for mid, _ in sims[:candidate_k]])

    # If no candidates, fall back to top popular movies
    if not candidates:
        popularity = df.groupby('movieId').size().sort_values(ascending=False)
        for mid in popularity.index:
            if mid not in rated_set:
                candidates.add(mid)
            if len(candidates) >= top_n * 5:
                break

    candidates = list(candidates)

    # 2️⃣ CF scoring
    user_tensor = torch.tensor([0]*len(candidates), dtype=torch.long).to(device)
    movie_tensor = torch.tensor([movie_map[m] for m in candidates], dtype=torch.long).to(device)
    with torch.no_grad():
        cf_scores = model(user_tensor, movie_tensor).detach().cpu().numpy()
    cf_scores = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min() + 1e-8)

    # 3️⃣ Content scoring
    content_scores = []
    for m in candidates:
        sims = [content_similarity(m_input, m) for m_input in input_movie_ids if m_input in movie_tfidf_idx]
        content_scores.append(np.mean(sims) if sims else 0)
    content_scores = np.array(content_scores)
    content_scores = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min() + 1e-8)

    # 4️⃣ Popularity penalty
    if pop_penalty:
        popularity = df.groupby('movieId').size().to_dict()
        pop_scores = np.array([1 / (np.log(1 + popularity.get(m,1))) for m in candidates])
        pop_scores = (pop_scores - pop_scores.min()) / (pop_scores.max() - pop_scores.min() + 1e-8)
    else:
        pop_scores = np.zeros(len(candidates))

    # 5️⃣ Hybrid scoring
    hybrid_scores = cf_weight * cf_scores + (1 - cf_weight) * content_scores + 0.1 * pop_scores
    top_idx = np.argsort(-hybrid_scores)[:top_n]

    recommendations = []
    for i in top_idx:
        movie_id = candidates[i]
        h_score = round(hybrid_scores[i], 3)
        sims = [(m_input, content_similarity(m_input, movie_id)) for m_input in input_movie_ids if m_input in movie_tfidf_idx]
        sims = [s for s in sims if s[1] > 0]
        sims.sort(key=lambda x: x[1], reverse=True)
        ref_strings = [f"{df[df['movieId']==m]['title'].values[0]} (sim={sim:.2f})" for m, sim in sims[:top_k_ref]]
        explanation = "Because you liked " + " and ".join(ref_strings) if ref_strings else "Recommended based on your preferences"
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

