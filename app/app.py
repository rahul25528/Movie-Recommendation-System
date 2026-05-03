"""
Movie Recommendation System — Memory Optimized for Render Free Tier
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import ast
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, template_folder='../templates', static_folder='../static')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'movies_metadata.csv')

print("Loading dataset (optimized)...")

COLS = ['title', 'overview', 'genres', 'vote_average', 'vote_count',
        'release_date', 'original_language']

df = pd.read_csv(DATA_PATH, usecols=COLS, low_memory=False)
df = df.dropna(subset=['title', 'overview']).reset_index(drop=True)

# Keep only English movies to reduce memory
df = df[df['original_language'] == 'en'].reset_index(drop=True)

# Keep only movies with at least 10 votes
df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce').fillna(0)
df = df[df['vote_count'] >= 10].reset_index(drop=True)

print(f"Working with {len(df):,} movies after filtering")

df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0)

def parse_names(obj_str):
    try:
        items = ast.literal_eval(obj_str)
        return ' '.join([x['name'] for x in items if 'name' in x])
    except:
        return ''

df['genres_clean'] = df['genres'].apply(parse_names)

# Weighted score
C = df['vote_average'].mean()
m = df['vote_count'].quantile(0.80)
df['weighted_score'] = df.apply(
    lambda r: (r['vote_count'] / (r['vote_count'] + m)) * r['vote_average'] +
              (m / (r['vote_count'] + m)) * C, axis=1)

# TF-IDF with reduced features to save memory
df['soup'] = df['overview'].fillna('') + ' ' + df['genres_clean'] + ' ' + df['genres_clean']
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['soup'])
indices = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()

# Free unused memory
df.drop(columns=['release_date', 'original_language', 'genres'], errors='ignore', inplace=True)
import gc; gc.collect()

print(f"✅ Ready — {len(df):,} movies loaded")


def movie_to_dict(row, extra=None):
    overview = str(row['overview'])
    d = {
        'title': row['title'],
        'year': int(row['year']) if pd.notna(row['year']) else 'N/A',
        'genres': row['genres_clean'],
        'rating': round(float(row['vote_average']), 1),
        'votes': int(row['vote_count']),
        'score': round(float(row['weighted_score']), 2),
        'overview': overview[:250] + '...' if len(overview) > 250 else overview
    }
    if extra:
        d.update(extra)
    return d

def get_idx(title_lower):
    iv = indices[title_lower]
    return int(iv.iloc[0]) if hasattr(iv, 'iloc') else int(iv)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/top')
def api_top():
    genre     = request.args.get('genre', '').strip()
    year_from = request.args.get('year_from', type=int)
    year_to   = request.args.get('year_to',   type=int)
    top_n     = min(int(request.args.get('top_n', 10)), 20)
    filtered  = df[df['vote_count'] >= m].copy()
    if genre:
        filtered = filtered[filtered['genres_clean'].str.contains(genre, case=False, na=False)]
    if year_from:
        filtered = filtered[filtered['year'] >= year_from]
    if year_to:
        filtered = filtered[filtered['year'] <= year_to]
    top = filtered.nlargest(top_n, 'weighted_score')
    return jsonify({'results': [movie_to_dict(r) for _, r in top.iterrows()], 'count': len(top)})

@app.route('/api/recommend')
def api_recommend():
    title  = request.args.get('title', '').strip()
    method = request.args.get('method', 'hybrid')
    top_n  = min(int(request.args.get('top_n', 10)), 20)
    if not title:
        return jsonify({'error': 'Please provide a movie title'}), 400
    title_lower = title.lower()
    if title_lower not in indices:
        matches = [t for t in indices.index if title_lower in t]
        if not matches:
            return jsonify({'error': f"'{title}' not found"}), 404
        title_lower = matches[0]
    idx = get_idx(title_lower)
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores[idx] = 0
    if method == 'content':
        top_idx = np.argsort(sim_scores)[::-1][:top_n]
        results = [movie_to_dict(df.iloc[i], {'similarity': round(float(sim_scores[i]), 3)}) for i in top_idx]
    else:
        sim_norm   = sim_scores / (sim_scores.max() + 1e-9)
        score_norm = (df['weighted_score'] - df['weighted_score'].min()) / \
                     (df['weighted_score'].max() - df['weighted_score'].min() + 1e-9)
        hybrid     = 0.6 * sim_norm + 0.4 * score_norm.values
        top_idx    = np.argsort(hybrid)[::-1][:top_n]
        results    = [movie_to_dict(df.iloc[i], {'hybrid_score': round(float(hybrid[i]), 4)}) for i in top_idx]
    return jsonify({'query': df.iloc[idx]['title'], 'method': method, 'results': results})

@app.route('/api/search')
def api_search():
    q = request.args.get('q', '').strip().lower()
    if len(q) < 2:
        return jsonify({'results': []})
    matches = df[df['title'].str.lower().str.contains(q, na=False)].head(8)
    results = [{'title': r['title'], 'year': int(r['year']) if pd.notna(r['year']) else ''} for _, r in matches.iterrows()]
    return jsonify({'results': results})

@app.route('/api/stats')
def api_stats():
    return jsonify({
        'total_movies': len(df),
        'avg_rating': round(float(C), 2),
        'min_votes_threshold': int(m),
        'year_range': [int(df['year'].min()), int(df['year'].max())],
        'languages': 1,
        'top_genres': df['genres_clean'].str.split().explode().value_counts().head(8).to_dict()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
