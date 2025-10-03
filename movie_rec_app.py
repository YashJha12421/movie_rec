import streamlit as st
import pandas as pd
import re
from rapidfuzz import process, fuzz
import torch
import pickle

# -----------------------------
# Load your model & mappings
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load NCF or hybrid model
model = torch.load("hybrid_diverse_model.pth", map_location=device)
model.eval()

# Load movie DataFrame
df = pd.read_csv("movies.csv")  # or wherever your movies data is

# Load saved user/movie maps if needed
with open("movie_map.pkl", "rb") as f:
    movie_map = pickle.load(f)

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
        best_match, score = process.extractOne(inp, movie_title_map.keys(), scorer=fuzz.token_sort_ratio)
        if score >= threshold:
            matched_ids.append(movie_title_map[best_match])
    return matched_ids

# -----------------------------
# Recommendation function
# -----------------------------
def recommend_from_list(movie_ids, top_n=10):
    # Call your existing hybrid model function here
    # Replace with whatever you already have:
    # recommend_hybrid_from_list(movie_ids)
    return recommend_hybrid_from_list(movie_ids, model=model, top_n=top_n, device=device)

# -----------------------------
# Streamlit App
# -----------------------------
st.title("🎬 Hybrid Movie Recommender")

user_input = st.text_area("Enter your favorite movies (comma-separated):")

if st.button("Recommend"):
    if not user_input.strip():
        st.warning("Please enter at least one movie.")
    else:
        input_movies = [m.strip() for m in user_input.split(",")]
        known_movie_ids = find_known_movies(input_movies, movie_title_map)
        
        if not known_movie_ids:
            st.warning("No known movies found from your input. Check spelling or try another movie.")
        else:
            recommendations = recommend_from_list(known_movie_ids)
            
            st.success(f"Found {len(recommendations)} recommendations!")
            for title, reason, score in recommendations:
                st.write(f"**{title}** — {reason} — Score: {score:.2f}")

