import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st

# Load the dataset (assuming 'tmdb_5000_movies.csv' contains your dataset)
# Replace 'path_to_your_dataset' with the actual path where you've stored the dataset
df = pd.read_csv('tmdb_5000_movies.csv', encoding='latin1')

# Preprocessing and feature extraction (you might need to adapt this part based on your dataset structure)
# For instance, concatenating relevant columns like genres, keywords, cast, etc.
features = ['genres', 'keywords']  # Replace with relevant columns
for feature in features:
    df[feature] = df[feature].fillna('')  # Replace missing values with empty string

# Combine features into a single column
df['combined_features'] = df['genres'] + ' ' + df['keywords']  # Adjust as needed

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Streamlit App
st.title('Movie Recommendation System')

# Movie input from user
movie_name = st.selectbox('Select a movie', df['title'].values)

# Recommendation function
def recommend_movies(movie_name, cosine_sim=cosine_sim, df=df):
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    idx = indices[movie_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Get top 5 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

# Display recommendations
if st.button('Get Recommendations'):
    if movie_name in df['title'].unique():
        recommendations = recommend_movies(movie_name)
        st.success('Top 5 Recommended Movies:')
        for movie in recommendations:
            st.write(movie)
    else:
        st.error('Movie not found in the dataset. Please enter another movie name.')
