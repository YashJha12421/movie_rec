import streamlit as st
import pandas as pd
import re
from rapidfuzz import process, fuzz
import torch
import pickle
from ncf_model import NCF

# -----------------------------
# Device setup
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------
# Load data & mappings
# -----------------------------
df = pd.read_csv("movies.csv")

with open("user_map.pkl", "rb") as f:
    user_map = pickle.load(f)
with open("movie_map.pkl", "rb") as f:
    movie_map = pickle.load(f)

num_users = len(user_map)
num_items = len(movie_map)

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
# Fuzzy match input
# -----------------------------
def find_known_movies(user_input_list, movie_title_map, threshold=60):
    matched_ids = []
    for inp in user_input_list:
        inp = inp.lower().strip()
        # rapidfuzz extractOne returns (match, score, index)
        best_match, score, _ = process.extractOne(inp, movie_title_map.keys(), scorer=fuzz.token_sort_ratio)
        if score >= threshold:
            matched_ids.append(movie_title_map[best_match])
    return matched_ids

# -----------------------------
# Recommendation function
# -----------------------------
def recommend_from_input(input_movie_ids, model, df, movie_map, top_n=10, device='cpu'):
    model.eval()
    
    all_movies = df['movieId'].values
    input_set = set(input_movie_ids)
    
    # Pseudo-user: dummy user for scoring
    pseudo_user_id = 0
    user_tensor = torch.tensor([pseudo_user_id]*len(all_movies), dtype=torch.long).to(device)
    movie_tensor = torch.tensor([movie_map[m] for m in all_movies], dtype=torch.long).to(device)
    
    # CF scoring
    with torch.no_grad():
        scores = model(user_tensor, movie_tensor).detach().cpu().numpy()
    
    # Top-N excluding input movies
    top_idx = np.argsort(-scores)
    recommendations = []
    for idx in top_idx:
        mid = all_movies[idx]
        if mid not in input_set:
            title = df[df['movieId']==mid]['title'].values[0]
            recommendations.append((title, "Recommended based on your input movies", scores[idx]))
        if len(recommendations) >= top_n:
            break
            
    return recommendations

# -----------------------------
# Streamlit App
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
            st.warning("No known movies found. Check spelling or try another movie.")
        else:
            recommendations = recommend_from_input(
                input_movie_ids=known_movie_ids,
                model=model,
                df=df,
                movie_map=movie_map,
                top_n=10,
                device=device
            )
            
            st.success(f"Found {len(recommendations)} recommendations!")
            for title, reason, score in recommendations:
                st.write(f"**{title}** — {reason} — Score: {score:.2f}")

            
            st.success(f"Found {len(recommendations)} recommendations!")
            for title, reason, score in recommendations:
                st.write(f"**{title}** — {reason} — Score: {score:.2f}")

