import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from datetime import datetime

# Download NLTK resources
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = joblib.load("imdb_sentiment_model.pkl")
vectorizer = joblib.load("imdb_vectorizer.pkl")

# ✅ Updated Hollywood posters with valid TMDB URLs
hollywood_movies = {
    "Inception": "https://image.tmdb.org/t/p/original/qmDpIHrmpJINaRKAfWQfftjCdyi.jpg",
    "The Dark Knight": "https://image.tmdb.org/t/p/original/1hRoyzDtpgMU7Dz4JF22RANzQO7.jpg",
    "Interstellar": "https://image.tmdb.org/t/p/original/rAiYTfKGqDCRIIqo664sY9XZIvQ.jpg",
    "Titanic": "https://image.tmdb.org/t/p/original/kHXEpyfl6zqn8a6YuozZUujufXf.jpg",
    "Avengers: Endgame": "https://image.tmdb.org/t/p/original/ulzhLuWrPK07P1YkdWQLZnQh1JL.jpg",
    "Joker": "https://image.tmdb.org/t/p/original/udDclJoHjfjb8Ekgsd4FDteOkCU.jpg",
    "Forrest Gump": "https://image.tmdb.org/t/p/original/saHP97rTPS5eLmrLQEcANmKrsFl.jpg",
    "Shutter Island": "https://image.tmdb.org/t/p/original/kve20tXwUZpu4GUX8l6X7Z4jmL6.jpg",
    "The Shawshank Redemption": "https://image.tmdb.org/t/p/original/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg",
    "Oppenheimer": "https://image.tmdb.org/t/p/original/ptpr0kGAckfQkJeJIt8st5dglvd.jpg"  # New
}

# ✅ Bollywood (unchanged but verified)
bollywood_movies = {
    "3 Idiots": "https://upload.wikimedia.org/wikipedia/en/d/df/3_idiots_poster.jpg",
    "Dangal": "https://upload.wikimedia.org/wikipedia/en/9/99/Dangal_Poster.jpg",
    "PK": "https://upload.wikimedia.org/wikipedia/en/c/c3/PK_poster.jpg",
    "Lagaan": "https://upload.wikimedia.org/wikipedia/en/b/b6/Lagaan.jpg",
    "Bahubali": "https://upload.wikimedia.org/wikipedia/en/5/5c/Baahubali_The_Beginning_poster.jpg",
    "Gully Boy": "https://upload.wikimedia.org/wikipedia/en/3/3f/Gully_Boy_poster.jpg",
    "Zindagi Na Milegi Dobara": "https://upload.wikimedia.org/wikipedia/en/a/a9/Zindaginamilegidobara.jpg",
    "Pathaan": "https://upload.wikimedia.org/wikipedia/en/c/c3/Pathaan_film_poster.jpg",
    "Shershaah": "https://upload.wikimedia.org/wikipedia/en/e/e7/Shershaah_film_poster.jpg",
    "Jawan": "https://upload.wikimedia.org/wikipedia/en/4/41/Jawan_film_poster.jpg"
}

# Clean text function
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    return " ".join([word for word in words if word not in stop_words])

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# Streamlit page config
st.set_page_config(page_title="IMDB Sentiment Analyzer", layout="centered", page_icon="🎬")

# Sidebar
with st.sidebar:
    st.title("🎬 IMDB Sentiment App")
    category = st.radio("Select Movie Category:", ["Hollywood", "Bollywood"])
    movies = hollywood_movies if category == "Hollywood" else bollywood_movies
    selected_movie = st.selectbox("🎥 Choose a Movie:", list(movies.keys()))
    st.markdown("""
    This app predicts if a movie review is **Positive** or **Negative**.

    - Select a movie  
    - Write your review  
    - Get prediction  
    - Give feedback 👍👎
    
    **Made with ❤️ by Anonymous Team**
    """)

# App title
st.markdown("<h1 style='text-align: center;'>IMDB Review Sentiment Analyzer</h1>", unsafe_allow_html=True)

# Movie poster
poster_url = movies.get(selected_movie)
if poster_url:
    try:
        st.image(poster_url, caption=f"{selected_movie} Poster", use_container_width=True)
    except:
        st.info("Poster could not be loaded.")
else:
    st.info("Poster not available.")

# Text input
review = st.text_area(f"Write your review for '{selected_movie}':", height=200)

# Predict
if st.button("🎯 Predict Sentiment"):
    if not review.strip():
        st.warning("Please enter a review before predicting.")
    else:
        cleaned = clean_text(review)
        vector = vectorizer.transform([cleaned]).toarray()
        prediction = model.predict(vector)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"

        # Show result
        if prediction == 1:
            st.success(f"🎉 Sentiment for '{selected_movie}': **Positive** 😊")
        else:
            st.error(f"😞 Sentiment for '{selected_movie}': **Negative**")

        # Save to history
        st.session_state.history.append({
            "movie": selected_movie,
            "review": review,
            "sentiment": sentiment,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # Thumbs up/down rating
        rating = st.radio("Was the prediction correct?", ["👍 Yes", "👎 No"])
        st.markdown(f"Thanks for your feedback: {rating}")

# History
if st.session_state.history:
    st.markdown("---")
    st.subheader("🕓 Prediction History")
    for entry in reversed(st.session_state.history):
        st.markdown(f"**{entry['time']}** — *{entry['movie']}* — **{entry['sentiment']}**")
        st.markdown(f"Review: {entry['review']}")
        st.markdown("---")
