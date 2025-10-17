# Movie Recommendation Engine

A hybrid recommendation system combining neural collaborative filtering with content-based filtering using the MovieLens 20M dataset (sampled to 100K ratings).

**[Live Demo](https://movierec-yj.streamlit.app/)** | [GitHub](https://github.com/YashJha12421/movie_rec)

## Quick Start

```bash
git clone https://github.com/YashJha12421/movie-recommendation-engine.git
pip install -r requirements.txt
streamlit run app.py
```

Or visit the live app at [movierec-yj.streamlit.app](https://movierec-yj.streamlit.app/)

## How It Works

1. **Data**: 100K ratings from MovieLens, stratified by movie popularity
2. **Model**: Neural Collaborative Filtering (NCF) with 64D embeddings trained using BPR loss
3. **Content**: TF-IDF genre vectors for similarity-based explanations
4. **Hybrid**: Combines CF scores (70%), content similarity (30%), and popularity penalty (10%)

The model learns latent user/item representations and ranks movies based on predicted user affinity, then explains recommendations by showing which of your selected movies influenced each suggestion.

## Features

- Select favorite movies → get top-10 personalized recommendations
- Explainable: Shows which movies from your history influenced each suggestion
- Diversity control: Penalizes popularity to surface niche films
- Fast inference: GPU-accelerated batch scoring

## Technical Details

- **Framework**: PyTorch
- **Training**: BPR loss with 10 negative samples per positive
- **Hardware**: Kaggle P100 GPU (~45 min training)
- **Model Size**: 12.3 MB
- **Dataset**: 36,745 users, 9,193 movies, 99.46% sparsity

## Limitations

- Cold start for new users (falls back to popularity)
- No temporal modeling (static 2015 snapshot)
- Retraining needed for new movies (no incremental learning)

## Author

Yash Jha | [GitHub](https://github.com/YashJha12421)
