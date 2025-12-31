import streamlit as st
import pandas as pd
import pickle
import os
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬"
)

st.title("🎬 Movie Recommendation System")
st.write("Select a movie to get similar movie recommendations.")

# ---------- Helper functions ----------
def convert(text):
    return [i["name"] for i in ast.literal_eval(text)]

def convert_cast(text):
    return [i["name"] for i in ast.literal_eval(text)[:3]]

def fetch_director(text):
    for i in ast.literal_eval(text):
        if i["job"] == "Director":
            return [i["name"]]
    return []

# ---------- Load or build model ----------
@st.cache_data(show_spinner=True)
def load_data_and_model():
    # If pickle files exist, load them
    if os.path.exists("movies.pkl") and os.path.exists("similarity.pkl"):
        movies = pickle.load(open("movies.pkl", "rb"))
        similarity = pickle.load(open("similarity.pkl", "rb"))
        return movies, similarity

    # Otherwise build from CSV files
    st.warning("Model files not found. Building model...")

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")


    movies = movies.merge(credits, on="title")
    movies = movies[["movie_id","title","overview","genres","keywords","cast","crew"]]
    movies.dropna(inplace=True)

    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"] = movies["cast"].apply(convert_cast)
    movies["crew"] = movies["crew"].apply(fetch_director)
    movies["overview"] = movies["overview"].apply(lambda x: x.split())

    for col in ["genres","keywords","cast","crew"]:
        movies[col] = movies[col].apply(lambda x: [i.replace(" ","") for i in x])

    movies["tags"] = (
        movies["overview"]
        + movies["genres"]
        + movies["keywords"]
        + movies["cast"]
        + movies["crew"]
    )

    movies["tags"] = movies["tags"].apply(lambda x: " ".join(x).lower())

    new_df = movies[["movie_id","title","tags"]]

    cv = CountVectorizer(max_features=3000, stop_words="english")
    vectors = cv.fit_transform(new_df["tags"]).toarray()
    similarity = cosine_similarity(vectors)

    # Save files for next run
    pickle.dump(new_df, open("movies.pkl", "wb"))
    pickle.dump(similarity, open("similarity.pkl", "wb"))

    return new_df, similarity

# ---------- Recommendation ----------
def recommend(movie):
    index = movies[movies["title"] == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    return [movies.iloc[i[0]].title for i in movie_list]

# ---------- Main ----------
try:
    movies, similarity = load_data_and_model()

    selected_movie = st.selectbox(
        "🎥 Choose a movie",
        movies["title"].values
    )

    if st.button("Recommend"):
        st.subheader("Recommended Movies:")
        for m in recommend(selected_movie):
            st.write("👉", m)

except Exception as e:
    st.error("Something went wrong. Please check dataset files.")
    st.code(str(e))
