# 🎬 Movie Recommendation System

A content-based Movie Recommendation System built using Machine Learning and deployed with Streamlit.

This project recommends similar movies based on genres, keywords, cast, crew, and overview using cosine similarity.

Streamlit App Link = https://movie-recommendation-fo4a258hl37zpyqjya43l3.streamlit.app/
---

## 🚀 Features
- Recommends top 5 similar movies
- Content-based filtering approach
- Interactive Streamlit web application
- Simple and easy-to-use interface

---

## 🛠️ Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

---

## 📊 Dataset
TMDB 5000 Movies Dataset (Kaggle)

Files used:
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

---

## ⚙️ Project Workflow
1. Data preprocessing and feature extraction
2. Text vectorization using CountVectorizer
3. Similarity calculation using cosine similarity
4. Model files saved using pickle
5. Streamlit app loads saved files and generates recommendations

---

## ▶️ How to Run the Project

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
Step 2: Generate Model Files

Run the Jupyter/Colab notebook to generate:

movies.pkl

similarity.pkl

Step 3: Run Streamlit App
streamlit run app.py

📁 Project Structure
movie-recommendation-system/

│

├── app.py

├── movie_recommender.ipynb

├── requirements.txt

├── README.md

├── .gitignore

└── data/
    ├── tmdb_5000_movies.csv
    └── tmdb_5000_credits.csv

📝 Note

Large generated files (movies.pkl, similarity.pkl) are excluded from the GitHub repository using .gitignore.
Please generate them locally by running the notebook before launching the app.

🎯 Future Improvements

Add movie posters using TMDB API

Improve recommendations using TF-IDF

Deploy the app on Streamlit Cloud
