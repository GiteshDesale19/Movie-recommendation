import streamlit as st
import pickle
import pandas as pd
import os

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Recommendation System")
st.write("Select a movie and get similar movie recommendations.")

# ---- Load files safely ----
if not os.path.exists("movies.pkl") or not os.path.exists("similarity.pkl"):
    st.error(
        "Required files not found.\n\n"
        "Please run the notebook first to generate:\n"
        "- movies.pkl\n"
        "- similarity.pkl"
    )
    st.stop()

movies = pickle.load(open("movies.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))

# ---- Recommendation function ----
def recommend(movie):
    movie_index = movies[movies["title"] == movie].index[0]
    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommendations = []
    for i in movie_list:
        recommendations.append(movies.iloc[i[0]].title)

    return recommendations

# ---- UI ----
selected_movie = st.selectbox(
    "🎥 Choose a movie",
    movies["title"].values
)

if st.button("Recommend"):
    st.subheader("Recommended Movies:")
    for movie in recommend(selected_movie):
        st.write("👉", movie)
