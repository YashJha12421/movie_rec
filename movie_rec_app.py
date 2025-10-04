import streamlit as st
import pandas as pd
import re
import torch
import pickle
import numpy as np
from ncf_model import NCF
from sklearn.metrics.pairwise import cosine_similarity

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
with open("tfidf.pkl", "rb") as f:
    movie_tfidf_idx = pickle.load(f)

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
        best_match = process.extractOne(inp, movie_title_map.keys(), scorer=fuzz.token_sort_ratio)
        if best_match and best_match[1] >= threshold:
            matched_ids.append(movie_title_map[best_match[0]])
    return matched_ids

# -----------------------------
# Content similarity
# -----------------------------
def content_similarity(movie_id1, movie_id2):
    if movie_id1 in movie_tfidf_idx and movie_id2 in movie_tfidf_idx:
        vec1 = movie_tfidf_idx[movie_id1]
        vec2 = movie_tfidf_idx[movie_id2]
        return cosine_similarity(vec1, vec2)[0,0]
    else:
        return 0

# -----------------------------
# Recommendation function
# -----------------------------
def recommend_hybrid(input_movie_ids, top_n=10, model=None, device='cpu', cf_weight=0.7, content_weight=0.3):
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

    # 3️⃣ Compute CF scores using NCF MLP
    user_emb_expanded = pseudo_user_emb.expand(len(all_ids), -1)
    item_embs = item_emb_matrix[all_ids_tensor].to(device)
    x = torch.cat([user_emb_expanded, item_embs], dim=-1)
    with torch.no_grad():
        cf_scores = model.mlp(x).squeeze().cpu().numpy()

    # 4️⃣ Compute content-based scores
    content_scores = []
    for m in all_ids:
        sims = [content_similarity(um, list(movie_map.keys())[list(movie_map.values()).index(m)]) for um in input_movie_ids]
        content_scores.append(np.mean(sims) if sims else 0)
    content_scores = np.array(content_scores)

    # 5️⃣ Weighted hybrid score
    hybrid_scores = cf_weight * cf_scores + content_weight * content_scores

    # 6️⃣ Top-N selection
    top_idx = np.argsort(-hybrid_scores)[:top_n]
    recommendations = []
    for i in top_idx:
        movie_id = list(movie_map.keys())[list(movie_map.values()).index(all_ids[i])]
        title = df[df['movieId'] == movie_id]['title'].values[0]

        # Build explanation: top 2 content matches
        sims = [(um, content_similarity(um, movie_id)) for um in input_movie_ids]
        sims.sort(key=lambda x: x[1], reverse=True)
        top_refs = sims[:2]
        explanation = "Because you liked " + " and ".join([df[df['movieId']==m]['title'].values[0] for m,_ in top_refs])
        
        recommendations.append((title, explanation, float(hybrid_scores[i])))

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
            recommendations = recommend_hybrid(known_movie_ids, top_n=10, model=model, device=device)
            
            st.success(f"Found {len(recommendations)} recommendations!")
            for title, reason, score in recommendations:
                st.write(f"**{title}** — {reason} — Score: {score:.2f}")

