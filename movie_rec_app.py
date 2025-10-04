# movie_rec_app.py
import streamlit as st
import pandas as pd
import re
import pickle
import joblib
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz
from ncf_model import NCF

# -----------------------------
# Configurable params
# -----------------------------
TOP_N = 10
CANDIDATE_K_PER_INPUT = 200   # how many candidates to consider per input movie (content-based)
CF_WEIGHT = 0.7               # weight for CF vs content
CONTENT_WEIGHT = 1.0 - CF_WEIGHT
POP_PENALTY_WEIGHT = 0.1
MMR_LAMBDA = 0.7              # diversity parameter (higher -> more relevance, lower -> more diverse)

# -----------------------------
# Device setup
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------
# Load data & artifacts (robust)
# -----------------------------
df = pd.read_csv("movies.csv")  # must contain movieId and title (and genres optional)

# load pickles
with open("movie_map.pkl", "rb") as f:
    movie_map = pickle.load(f)   # expected: mapping movieId -> internal model index
with open("user_map.pkl", "rb") as f:
    user_map = pickle.load(f)

# TF-IDF artifacts: we expect a matrix (genre_matrix) and a mapping movieId->row_idx
# Common filenames: "tfidf.pkl" for matrix and "movie_tfidf_idx.pkl" for mapping
# We'll try a few names to be robust.
def safe_load_tfidf():
    # try joblib load for matrix
    tried = []
    for name in ("tfidf.pkl", "movie_tfidf_idx.pkl", "movie_tfidf.pkl", "genre_tfidf.pkl"):
        tried.append(name)
    # load matrix
    try:
        genre_matrix = joblib.load("tfidf.pkl")  # expected matrix (n_movies x features)
        movie_tfidf_idx = joblib.load("movie_tfidf_idx.pkl")
        return genre_matrix, movie_tfidf_idx
    except Exception:
        # try fallback: maybe the saved names are swapped
        try:
            movie_tfidf_idx = joblib.load("movie_tfidf_idx.pkl")
            genre_matrix = joblib.load("tfidf.pkl")
            return genre_matrix, movie_tfidf_idx
        except Exception as e:
            st.error(f"Failed loading TF-IDF artifacts: {e}")
            raise

genre_matrix, movie_tfidf_idx = safe_load_tfidf()

# Build reverse mapping idx -> movieId for tfidf matrix rows
movieId_from_tfidf_idx = {v: k for k, v in movie_tfidf_idx.items()}

# -----------------------------
# Load NCF model
# -----------------------------
num_users = len(user_map)
num_items = len(movie_map)
model = NCF(num_users, num_items).to(device)
state = torch.load("ncf_model.pth", map_location=device)
# If it's a state_dict, load_state_dict; if it's an entire model, handle both:
if isinstance(state, dict) and not any(k.startswith('module') and isinstance(state[k], torch.Tensor) for k in state):
    # assume state is state_dict
    try:
        model.load_state_dict(state)
    except Exception:
        # if they saved a dict with extra keys, try common 'model_state_dict' key
        if 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
else:
    # state might be a whole model (rare here)
    try:
        model = state.to(device)
    except Exception:
        model.load_state_dict(state)

model.eval()

# -----------------------------
# Helpers: preprocessing & fuzzy matching
# -----------------------------
def normalize_title(title):
    return re.sub(r'\(\d{4}\)', '', title).strip().lower()

movie_title_map = {normalize_title(title): mid for mid, title in zip(df['movieId'], df['title'])}

def find_known_movies(user_input_list, movie_title_map, threshold=60):
    matched_ids = []
    for inp in user_input_list:
        inp_norm = inp.lower().strip()
        best = process.extractOne(inp_norm, list(movie_title_map.keys()), scorer=fuzz.token_sort_ratio)
        if best:
            match_title, score, _ = best
            if score >= threshold:
                matched_ids.append(movie_title_map[match_title])
    # dedupe preserving order
    seen = set()
    res = []
    for m in matched_ids:
        if m not in seen:
            seen.add(m)
            res.append(m)
    return res

# -----------------------------
# Content similarity util
# -----------------------------
def content_similarity(mid1, mid2):
    # Both mids must be in movie_tfidf_idx
    if mid1 not in movie_tfidf_idx or mid2 not in movie_tfidf_idx:
        return 0.0
    idx1 = movie_tfidf_idx[mid1]
    idx2 = movie_tfidf_idx[mid2]
    # genre_matrix row access (sparse or dense)
    v1 = genre_matrix[idx1]
    v2 = genre_matrix[idx2]
    return float(cosine_similarity(v1, v2)[0, 0])

# Fast top-K retrieval for a given input movie (content)
def top_k_candidates_by_content(input_mid, k=CANDIDATE_K_PER_INPUT):
    if input_mid not in movie_tfidf_idx:
        return []
    idx = movie_tfidf_idx[input_mid]
    vec = genre_matrix[idx]
    sims = cosine_similarity(vec, genre_matrix).ravel()  # vectorized
    # get indices of top k (excluding itself)
    # argsort descending
    top_idx = np.argpartition(-sims, range(min(k+1, sims.shape[0])))[:k+1]
    # sort them properly
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    results = []
    for tidx in top_idx:
        mid = movieId_from_tfidf_idx.get(int(tidx))
        if mid is None or mid == input_mid:
            continue
        results.append((mid, float(sims[tidx])))
        if len(results) >= k:
            break
    return results

# -----------------------------
# Popularity precompute
# -----------------------------
pop_counts = df.groupby('movieId').size().to_dict()
# normalize pop for penalty use later
pop_array_example = np.array(list(pop_counts.values())) if pop_counts else np.array([1.0])
pop_min = pop_array_example.min()
pop_max = pop_array_example.max()

def pop_penalty_score(mid):
    c = pop_counts.get(mid, 1)
    # higher c -> smaller penalty, so we invert log
    return 1.0 / (np.log(1 + c) + 1e-8)

# -----------------------------
# Main hybrid recommender (robust + diverse)
# -----------------------------
def recommend_hybrid(input_movie_ids, top_n=TOP_N, cf_weight=CF_WEIGHT, candidate_k=CANDIDATE_K_PER_INPUT, mmr_lambda=MMR_LAMBDA):
    """
    input_movie_ids: list of movieIds (as in df['movieId'])
    returns: list of (title, explanation, score)
    """
    if not input_movie_ids:
        return []

    # 1) candidate generation via content (top k per input)
    candidate_set = set()
    for mid in input_movie_ids:
        cand_list = top_k_candidates_by_content(mid, k=candidate_k)
        for m, s in cand_list:
            candidate_set.add(m)

    # fallback: if no candidates (weird), use top-popular movies
    if not candidate_set:
        sorted_pop = sorted(pop_counts.items(), key=lambda x: x[1], reverse=True)
        for mid, _ in sorted_pop:
            if mid not in input_movie_ids:
                candidate_set.add(mid)
            if len(candidate_set) >= top_n * 10:
                break

    # turn into list
    candidates = list(candidate_set)

    # 2) build pseudo-user embedding from input movies
    # map input mids to model indices using movie_map
    input_model_idxs = []
    for m in input_movie_ids:
        if m in movie_map:
            input_model_idxs.append(movie_map[m])
    if not input_model_idxs:
        # if none of input movies map to model indices, fallback to popular-based recommendations
        candidates = list(df['movieId'].values)[:top_n]
        return [(df[df['movieId']==mid]['title'].values[0], "Popular fallback", 0.0) for mid in candidates[:top_n]]

    with torch.no_grad():
        # item embedding matrix
        item_emb_matrix = model.item_emb.weight  # (num_items, emb_size)
        # gather input embeddings (ensure device)
        input_embs = item_emb_matrix[torch.tensor(input_model_idxs, dtype=torch.long)].to(device)
        pseudo_user_emb = input_embs.mean(dim=0, keepdim=True)  # shape (1, emb_size)

        # candidate model indices (some candidates may not be in movie_map, filter)
        cand_model_idxs = []
        cand_movie_ids = []
        for m in candidates:
            if m in movie_map:
                cand_model_idxs.append(movie_map[m])
                cand_movie_ids.append(m)
        if not cand_model_idxs:
            return []

        cand_idxs_tensor = torch.tensor(cand_model_idxs, dtype=torch.long, device=device)
        item_embs = item_emb_matrix[cand_idxs_tensor].to(device)

        # Build MLP input like during training: concat user_emb and item_emb
        user_expanded = pseudo_user_emb.expand(len(cand_model_idxs), -1)
        mlp_input = torch.cat([user_expanded, item_embs], dim=-1)  # (num_candidates, 2*emb)
        scores_tensor = model.mlp(mlp_input).squeeze()  # (num_candidates,)
        cf_scores = scores_tensor.detach().cpu().numpy()

    # 3) content scores for candidates (mean similarity to input movies)
    content_scores = []
    for m in cand_movie_ids:
        sims = []
        for im in input_movie_ids:
            if im in movie_tfidf_idx and m in movie_tfidf_idx:
                sims.append(content_similarity(im, m))
        content_scores.append(np.mean(sims) if sims else 0.0)
    content_scores = np.array(content_scores)

    # 4) normalize CF and content scores to [0,1]
    def normalize(arr):
        arr = np.array(arr, dtype=float)
        if arr.size == 0:
            return arr
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-8:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn + 1e-12)

    cf_norm = normalize(cf_scores)
    content_norm = normalize(content_scores)
    pop_norm = np.array([pop_penalty_score(m) for m in cand_movie_ids])
    pop_norm = normalize(pop_norm)

    # 5) hybrid score
    hybrid_scores = cf_weight * cf_norm + (1.0 - cf_weight) * content_norm + POP_PENALTY_WEIGHT * pop_norm

    # 6) MMR-style diversity selection:
    selected = []
    selected_scores = []
    candidate_indices = list(range(len(cand_movie_ids)))
    # sort candidates by hybrid score descending
    sorted_idx = list(np.argsort(-hybrid_scores))
    # initialize list of remaining indices
    remaining = sorted_idx.copy()
    while remaining and len(selected) < top_n:
        if not selected:
            # pick top
            idx = remaining.pop(0)
            selected.append(idx)
            selected_scores.append(hybrid_scores[idx])
            continue
        # compute MMR score for each remaining candidate
        mmr_vals = []
        for idx in remaining:
            relevance = hybrid_scores[idx]
            # compute max similarity to already selected (content similarity used)
            sims_to_selected = []
            for sidx in selected:
                m1 = cand_movie_ids[idx]
                m2 = cand_movie_ids[sidx]
                if m1 in movie_tfidf_idx and m2 in movie_tfidf_idx:
                    sims_to_selected.append(content_similarity(m1, m2))
            max_sim = max(sims_to_selected) if sims_to_selected else 0.0
            mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * max_sim
            mmr_vals.append((idx, mmr_score))
        # pick best mmr
        mmr_vals.sort(key=lambda x: -x[1])
        best_idx = mmr_vals[0][0]
        remaining.remove(best_idx)
        selected.append(best_idx)
        selected_scores.append(hybrid_scores[best_idx])

    # 7) build output
    recs = []
    for idx in selected:
        mid = cand_movie_ids[idx]
        title = df[df['movieId'] == mid]['title'].values[0]
        # explanation: top contributing input movies by content similarity
        ref_sims = []
        for im in input_movie_ids:
            if im in movie_tfidf_idx and mid in movie_tfidf_idx:
                ref_sims.append((im, content_similarity(im, mid)))
        ref_sims.sort(key=lambda x: -x[1])
        refs = [df[df['movieId'] == r[0]]['title'].values[0] + f" (sim={r[1]:.2f})" for r in ref_sims[:2]]
        explanation = "Because you liked " + " and ".join(refs) if refs else "Recommended based on your input movies"
        recs.append((title, explanation, float(hybrid_scores[idx])))

    return recs

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎬 Hybrid Movie Recommender (Hybrid: CF + Content + Diversity)")

user_input = st.text_area("Enter your favorite movies (comma-separated):", key="full_app_input")

if st.button("Recommend"):
    if not user_input.strip():
        st.warning("Please enter at least one movie.")
    else:
        inputs = [s.strip() for s in user_input.split(",") if s.strip()]
        known_ids = find_known_movies(inputs, movie_title_map, threshold=60)
        if not known_ids:
            st.warning("No known movies found. Try different spellings or include full titles.")
        else:
            with st.spinner("Computing recommendations..."):
                recs = recommend_hybrid(known_ids, top_n=TOP_N)
            if not recs:
                st.info("No recommendations found (edge case).")
            else:
                st.success(f"Found {len(recs)} recommendations!")
                for title, explanation, score in recs:
                    st.write(f"**{title}** — {explanation} — Score: {score:.3f}")
