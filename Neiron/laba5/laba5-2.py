import pandas as pd
import numpy as np
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_random_state
import warnings

warnings.filterwarnings("ignore")
ratings = pd.read_csv("ratings_small.csv")
movies = pd.read_csv("movies_metadata.csv", low_memory=False)

print(f"Всего оценок: {len(ratings)}")
print(f"Уникальных пользователей: {ratings['userId'].nunique()}")
print(f"Уникальных фильмов: {ratings['movieId'].nunique()}")

ratings = ratings.dropna(subset=['userId', 'movieId', 'rating'])
movies = movies.dropna(subset=['id', 'title'])
movies['id'] = movies['id'].astype(str)

user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
X = user_movie_matrix.values
movie_ids = user_movie_matrix.columns

def preprocess_data(X, binary=False):
    if binary:
        return (X > 0).astype(float)
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
        return scaler.fit_transform(X)

X_binary = preprocess_data(X, binary=True)
rbm = BernoulliRBM(n_components=100, learning_rate=0.01, n_iter=20, verbose=True, random_state=42)
rbm.fit(X_binary)

def predict_ratings(rbm, user_ratings, n_gibbs=10):
    rng = check_random_state(rbm.random_state)
    hidden = rbm._sample_hiddens(user_ratings, rng)

    for _ in range(n_gibbs):
        visible = rbm._sample_visibles(hidden, rng)
        hidden = rbm._sample_hiddens(visible, rng)

    return visible

def recommend_movies(user_id, n_recommendations=10, binary=True):
    user_index = user_id - 1
    user_ratings = X[user_index]
    user_reconstructed = predict_ratings(rbm, user_ratings, n_gibbs=10)
    unseen = np.where(user_ratings == 0)[0]
    recommended_indices = unseen[np.argsort(user_reconstructed[unseen])[::-1][:n_recommendations]]
    recommended_movie_ids = movie_ids[recommended_indices].astype(str)
    recommended_titles = movies[movies['id'].isin(recommended_movie_ids)]['title'].unique()

    return recommended_titles

user_id = 135
recommended_titles = recommend_movies(user_id, n_recommendations=10, binary=True)

print(f"\nРекомендованные фильмы для пользователя {user_id}:\n")
for i, title in enumerate(recommended_titles, 1):
    print(f"{i}. {title}")
