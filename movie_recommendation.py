# ============================================================
# MOVIE RECOMMENDATION SYSTEM
# Dataset: movies_metadata.csv (45,466 movies)
# Techniques: Content-Based + Weighted Rating (IMDB Formula)
# ============================================================

import pandas as pd
import numpy as np
import ast
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD & CLEAN DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("MOVIE RECOMMENDATION SYSTEM")
print("=" * 60)

df = pd.read_csv('/mnt/user-data/uploads/movies_metadata.csv', low_memory=False)
print(f"\n✅ Dataset loaded: {df.shape[0]:,} movies, {df.shape[1]} features")

# Drop rows with no title or overview
df = df.dropna(subset=['title', 'overview'])
df = df.reset_index(drop=True)

# Parse release year
df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year

# Parse genres from JSON-like string
def parse_names(obj_str):
    try:
        items = ast.literal_eval(obj_str)
        return ' '.join([x['name'] for x in items if 'name' in x])
    except:
        return ''

df['genres_clean'] = df['genres'].apply(parse_names)

# Parse spoken languages
df['languages_clean'] = df['spoken_languages'].apply(parse_names)

# Numeric conversions
df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0)
df['vote_count']   = pd.to_numeric(df['vote_count'],   errors='coerce').fillna(0)
df['popularity']   = pd.to_numeric(df['popularity'],   errors='coerce').fillna(0)

print(f"✅ After cleaning: {len(df):,} movies retained")


# ─────────────────────────────────────────────
# 2. WEIGHTED RATING (IMDB Formula)
# ─────────────────────────────────────────────
C  = df['vote_average'].mean()          # mean rating across all movies
m  = df['vote_count'].quantile(0.80)    # minimum votes required (80th percentile)

def weighted_rating(row, m=m, C=C):
    v = row['vote_count']
    R = row['vote_average']
    return (v / (v + m)) * R + (m / (v + m)) * C

df['weighted_score'] = df.apply(weighted_rating, axis=1)
print(f"\n✅ Weighted scores computed (C={C:.2f}, min_votes={m:.0f})")


# ─────────────────────────────────────────────
# 3. CONTENT-BASED FILTERING
#    (TF-IDF on overview + genres)
# ─────────────────────────────────────────────
df['soup'] = (
    df['overview'].fillna('') + ' ' +
    df['genres_clean'] + ' ' +
    df['genres_clean']     # genres weighted 2x
)

tfidf = TfidfVectorizer(stop_words='english', max_features=15000, ngram_range=(1,2))
tfidf_matrix = tfidf.fit_transform(df['soup'])
print(f"✅ TF-IDF matrix built: {tfidf_matrix.shape}")

# Build reverse-lookup: movie title → index
indices = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()


# ─────────────────────────────────────────────
# 4. RECOMMENDATION FUNCTIONS
# ─────────────────────────────────────────────

def get_content_recommendations(title, top_n=10):
    """Content-based: find similar movies by plot & genre."""
    title_lower = title.lower().strip()
    if title_lower not in indices:
        # Fuzzy fallback
        matches = [t for t in indices.index if title_lower in t]
        if not matches:
            return f"❌ '{title}' not found. Try a different title."
        title_lower = matches[0]
        print(f"  (Closest match found: '{df.loc[indices[title_lower], 'title']}')")

    idx_val = indices[title_lower]
    idx = int(idx_val.iloc[0]) if hasattr(idx_val, 'iloc') else int(idx_val)
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores[idx] = 0  # exclude self
    top_idx = np.argsort(sim_scores)[::-1][:top_n]
    top_idx = [i for i in top_idx if i < len(df)]

    results = df.iloc[top_idx][['title', 'genres_clean', 'year', 'vote_average', 'weighted_score', 'overview']].copy()
    results['similarity'] = sim_scores[top_idx].round(3)
    results['year'] = results['year'].fillna(0).astype(int)
    return results.reset_index(drop=True)


def get_top_movies(genre=None, year_from=None, year_to=None, top_n=10, min_votes=None):
    """Chart-based: top rated movies with optional filters."""
    mv = m if min_votes is None else min_votes
    filtered = df[df['vote_count'] >= mv].copy()

    if genre:
        filtered = filtered[filtered['genres_clean'].str.contains(genre, case=False, na=False)]
    if year_from:
        filtered = filtered[filtered['year'] >= year_from]
    if year_to:
        filtered = filtered[filtered['year'] <= year_to]

    top = filtered.nlargest(top_n, 'weighted_score')[
        ['title', 'genres_clean', 'year', 'vote_average', 'vote_count', 'weighted_score']
    ].reset_index(drop=True)
    top['year'] = top['year'].fillna(0).astype(int)
    top.index += 1
    return top


def get_hybrid_recommendations(title, top_n=10):
    """Hybrid: content similarity × weighted score."""
    title_lower = title.lower().strip()
    if title_lower not in indices:
        matches = [t for t in indices.index if title_lower in t]
        if not matches:
            return f"❌ '{title}' not found."
        title_lower = matches[0]

    idx_val = indices[title_lower]
    idx = int(idx_val.iloc[0]) if hasattr(idx_val, 'iloc') else int(idx_val)
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores[idx] = 0

    # Normalize both signals to [0,1]
    sim_norm  = sim_scores / (sim_scores.max() + 1e-9)
    score_norm = (df['weighted_score'] - df['weighted_score'].min()) / \
                 (df['weighted_score'].max() - df['weighted_score'].min() + 1e-9)

    hybrid = 0.6 * sim_norm + 0.4 * score_norm.values
    top_idx = np.argsort(hybrid)[::-1][:top_n]

    results = df.iloc[top_idx][['title', 'genres_clean', 'year', 'vote_average', 'weighted_score', 'overview']].copy()
    results['hybrid_score'] = hybrid[top_idx].round(4)
    results['year'] = results['year'].fillna(0).astype(int)
    return results.reset_index(drop=True)


# ─────────────────────────────────────────────
# 5. DEMO OUTPUT
# ─────────────────────────────────────────────

pd.set_option('display.max_columns', 8)
pd.set_option('display.width', 120)
pd.set_option('display.max_colwidth', 40)

print("\n" + "=" * 60)
print("📊 TOP 10 MOVIES OF ALL TIME (Weighted Rating)")
print("=" * 60)
print(get_top_movies(top_n=10).to_string())

print("\n" + "=" * 60)
print("🎬 TOP 10 ACTION MOVIES (2000–2020)")
print("=" * 60)
print(get_top_movies(genre='Action', year_from=2000, year_to=2020, top_n=10).to_string())

print("\n" + "=" * 60)
print("🔍 CONTENT-BASED RECOMMENDATIONS for 'The Dark Knight'")
print("=" * 60)
recs = get_content_recommendations('The Dark Knight', top_n=10)
print(recs[['title','genres_clean','year','vote_average','similarity']].to_string())

print("\n" + "=" * 60)
print("⚡ HYBRID RECOMMENDATIONS for 'Toy Story'")
print("=" * 60)
recs2 = get_hybrid_recommendations('Toy Story', top_n=10)
print(recs2[['title','genres_clean','year','vote_average','hybrid_score']].to_string())

print("\n" + "=" * 60)
print("✅ Recommendation system ready!")
print("=" * 60)
