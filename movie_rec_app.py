# movie_rec_app.py
import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np
from ncf_model import NCF

# --------------------------
# Load user/movie maps
# --------------------------
with open("user_map.pkl", "rb") as f:
    user_map = pickle.load(f)
with open("movie_map.pkl", "rb") as f:
    movie_map = pickle.load(f)

num_users = len(user_map)
num_movies = len(movie_map)

# --------------------------
# Load trained model
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NCF(num_users, num_movies)
model.load_state_dict(torch.load("ncf_model.pth", map_location=device))
model.to(device)
model.eval()

# --------------------------
# Helper: recommend function
# --------------------------
def recommend(user_movies, top_n=10):
    # Map input movies to IDs, ignore unknowns
    input_ids = [movie_map[movie] for movie in user_movies if movie in movie_map]
    if not input_ids:
        return []

    # Compute scores for all movies
    all_movie_ids = list(range(num_movies))
    user_ids = torch.tensor([0]*num_movies).to(device)  # dummy user 0
    movie_ids = torch.tensor(all_movie_ids).to(device)

    with torch.no_grad():
        scores = model(user_ids, movie_ids).cpu().numpy()

    # Remove movies already rated
    for mid in input_ids:
        scores[mid] = -np.inf

    top_indices = np.argsort(-scores)[:top_n]
    # Map back to movie names
    inv_movie_map = {v: k for k, v in movie_map.items()}
    recommendations = [inv_movie_map[i] for i in top_indices]
    return recommendations

# --------------------------
# Streamlit UI
# --------------------------
st.title("Movie Recommendation App 🎬")

user_input = st.text_area(
    "Enter movies you liked (comma separated):",
    "The Matrix (1999), Inception (2010)"
)

if st.button("Recommend"):
    movies = [m.strip() for m in user_input.split(",")]
    recs = recommend(movies)
    if recs:
        st.subheader("Top Recommendations:")
        for i, r in enumerate(recs, 1):
            st.write(f"{i}. {r}")
    else:
        st.write("No known movies found from your input.")
