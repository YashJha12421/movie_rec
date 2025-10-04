import streamlit as st
import pandas as pd
import re
import torch
import pickle
from ncf_model import NCF
import numpy as np

# -----------------------------
# Device
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------
# Load data & mappings
# -----------------------------
df = pd.read_csv("movies.csv")

with open("movie_map.pkl", "rb") as f:
    movie_map = pickle.load(f)
with open("user_map.pkl", "rb") as f:
    user_map = pickle.load(f)

num_users = len(user_map)
num_items = len(movie_map)

# -----------------------------
# Load model
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
# Fuzzy match input
# -----------------------------
from rapidfuzz import process, fuzz

def find_known_movies(user_input_list, movie_title_map, threshold=60):
    matched_ids = []
    for inp in user_input_list:
        inp = inp.lower().strip()
        # extractOne now returns (match, score, index)
        best_match = process.extractOne(inp, movie_title_map.keys(), scorer=fuzz.token_sort_ratio)
        if best_match and best_match[1] >= threshold:
            matched_ids.append(movie_title_map[best_match[0]])
    return matched_ids

# -----------------------------
# Recommendation function
# -----------------------------
def recommend_from_movies(input_movie_ids, top_n=10, model=None, device='cpu'):
    if not input_movie_ids:
        return []

    model.eval()
    
    # 1️⃣ Get input movie embeddings
    with torch.no_grad():
        item_emb_matrix = model.item_emb.weight  # (num_items, emb_size)
        input_embs = torch.stack([item_emb_matrix[movie_map[m]].to(device) for m in input_movie_ids])
        pseudo_user_emb = input_embs.mean(dim=0, keepdim=True)  # shape: (1, emb_size)

    # 2️⃣ Candidate movies (exclude input)
    all_ids = [m for m in movie_map.values() if m not in [movie_map[m] for m in input_movie_ids]]
    all_ids_tensor = torch.tensor(all_ids, dtype=torch.long, device=device)

    # 3️⃣ Compute scores using dot with pseudo-user embedding + MLP
    user_emb_expanded = pseudo_user_emb.expand(len(all_ids), -1)  # (num_candidates, emb_size)
    item_embs = item_emb_matrix[all_ids_tensor].to(device)
    x = torch.cat([user_emb_expanded, item_embs], dim=-1)
    with torch.no_grad():
        scores = model.mlp(x).squeeze().cpu().numpy()

    # 4️⃣ Top-N movies
    top_idx = np.argsort(-scores)[:top_n]
    recommendations = []
    for i in top_idx:
        movie_id = list(movie_map.keys())[list(movie_map.values()).index(all_ids[i])]
        title = df[df['movieId'] == movie_id]['title'].values[0]
        explanation = f"Recommended based on your input movies"
        recommendations.append((title, explanation, float(scores[i])))

    return recommendations

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎬 Hybrid Movie Recommender")

user_input = st.text_area("Enter your favorite movies (comma-separated):", key="movie_input")

if st.button("Recommend"):
    if not user_input.strip():
        st.warning("Please enter at least one movie.")
    else:
        input_movies = [m.strip() for m in user_input.split(",")]
        known_movie_ids = find_known_movies(input_movies, movie_title_map)
        
        if not known_movie_ids:
            st.warning("No known movies found from your input. Check spelling or try another movie.")
        else:
            recommendations = recommend_from_movies(known_movie_ids, top_n=10, model=model, device=device)
            
            st.success(f"Found {len(recommendations)} recommendations!")
            for title, reason, score in recommendations:
                st.write(f"**{title}** — {reason} — Score: {score:.2f}")


