import streamlit as st
import pandas as pd
import pickle
import os
import ast
import glob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬")

st.title("🎬 Movie Recommendation System")
st.write("Select a movie and get similar movie recommendations.")

# ---------- DEBUG ----------
st.write("📂 Files available to app:", os.listdir("."))

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
@st.cache_data
def load_model():

    # Load cached model if exists
    if os.path.exists("movies.pkl") and os.path.exists("similarity.pkl"):
        return (
            pickle.load(open("movies.pkl", "rb")),
            pickle.load(open("similarity.pkl", "rb"))
        )

    st.info("Model files not found. Building model...")

    # Auto-detect CSVs
    movie_files = glob.glob("*movie*.csv")
    credit_files = glob.glob("*credit*.csv")

    if not movie_files or not credit_files:
        st.error("CSV files not found. Please upload movie & credit CSV files.")
        st.stop()

    movie_file = movie_files[0]
    credit_file = credit_files[0]

    st.write("🎬 Movie file detected:", movie_file)
    st.write("🎭 Credit file detected:", credit_file)

    movies = pd.read_csv(movie_file)
    credits = pd.read_csv(credit_file)

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

    pickle.dump(new_df, open("movies.pkl","wb"))
    pickle.dump(similarity, open("similarity.pkl","wb"))

    return new_df, similarity

# ---------- App ----------
movies, similarity = load_model()

def recommend(movie):
    idx = movies[movies["title"] == movie].index[0]
    distances = similarity[idx]
    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]
    return [movies.iloc[i[0]].title for i in movie_list]

selected_movie = st.selectbox("🎥 Choose a movie", movies["title"].values)

if st.button("Recommend"):
    st.subheader("Recommended Movies")
    for m in recommend(selected_movie):
        st.write("👉", m)
