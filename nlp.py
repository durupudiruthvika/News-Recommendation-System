import streamlit as st
import hashlib
import json
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk
from googletrans import Translator

nltk.download('stopwords')

USER_DB_FILE = "user_credentials.json"
API_KEY = "  "

GENRES = {
    "en": ["Technology", "Sports", "Politics", "Entertainment", "Health", "Business"],
    "hi": ["प्रौद्योगिकी", "खेल", "राजनीति", "मनोरंजन", "स्वास्थ्य", "व्यापार"],  # Hindi
    "bn": ["প্রযুক্তি", "খেলা", "রাজনীতি", "বিনোদন", "স্বাস্থ্য", "ব্যবসা"],  # Bengali
    "ta": ["தொழில்நுட்பம்", "விளையாட்டு", "அரசியல்", "வினோதம்", "ஆரோக்கியம்", "வணிகம்"],  # Tamil
    "te": ["సాంకేతికత", "క్రీడలు", "రాజకీయాలు", "వినోదం", "ఆరోగ్యం", "వ్యాపారం"],  # Telugu
    "mr": ["तंत्रज्ञान", "क्रीडा", "राजकारण", "मनोरंजन", "आरोग्य", "व्यवसाय"],  # Marathi
    "gu": ["ટેક્નોલોજી", "ક્રીડાઓ", "રાજકારણ", "મનોરંજન", "આરોગ્ય", "વ્યાપાર"],  # Gujarati
    "kn": ["ತಂತ್ರಜ್ಞಾನ", "ಕ್ರೀಡೆ", "ರಾಜಕೀಯ", "ಮೂಡಲ", "ಆರೋಗ್ಯ", "ವ್ಯಾಪಾರ"],  # Kannada
    "ml": ["ടെക്‌നോളജി", "വൈദ്യുതി", "രാഷ്ട്രনীতি", "ആശംസ", "ആരോഗ്യസംരക്ഷണം", "വ്യാപാരം"],  # Malayalam
}

LANGUAGES = ["en", "hi", "bn", "ta", "te", "mr", "gu", "kn", "ml"]  # Language codes
translator = Translator()

# Custom CSS to improve the theme and layout
st.markdown("""
    <style>
        body {
            background-color: #f4f4f9;
            color: #333333;
        }
        .stButton>button {
            background-color: #1f77b4;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            height: 40px;
            width: 150px;
        }
        .stTextInput input {
            border: 2px solid #1f77b4;
            border-radius: 8px;
            padding: 10px;
        }
        .stSelectbox select {
            border: 2px solid #1f77b4;
            border-radius: 8px;
            padding: 10px;
            background-color: #ffffff;
        }
        .stTitle {
            font-size: 32px;
            font-weight: bold;
            color: #1f77b4;
        }
        .stSubheader {
            font-size: 24px;
            color: #1f77b4;
        }
        .stMarkdown {
            font-size: 16px;
            color: #333333;
        }
    </style>
""", unsafe_allow_html=True)

def load_user_db():
    try:
        with open(USER_DB_FILE, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def save_user_db(user_db):
    with open(USER_DB_FILE, "w") as file:
        json.dump(user_db, file)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username, password):
    user_db = load_user_db()
    if username in user_db and user_db[username] == hash_password(password):
        return True
    return False

def register_user(username, password, lang):
    user_db = load_user_db()
    if username in user_db:
        return False
    user_db[username] = {"password": hash_password(password), "language": lang}
    save_user_db(user_db)
    return True

def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.lower().split() if word.isalpha() and word not in stop_words])

def extract_features(articles):
    vectorizer = TfidfVectorizer(max_features=5000)
    texts = [preprocess_text(article.get("snippet", "")) for article in articles]
    texts = [text for text in texts if text.strip()]
    if not texts:
        raise ValueError("No valid articles to process after preprocessing.")
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer

def fetch_news(api_key, query):
    url = f"https://serpapi.com/search.json?q={query}&tbm=nws&api_key={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        raise ConnectionError("Failed to fetch news articles.")
    data = response.json()
    return data.get("news_results", [])

def recommend_articles(index, tfidf_matrix, articles, num_recommendations=10):
    similarity = cosine_similarity(tfidf_matrix[index], tfidf_matrix).flatten()
    similar_indices = similarity.argsort()[-(num_recommendations + 1):-1][::-1]
    recommendations = [{"title": articles[i]["title"], "link": articles[i]["link"], "snippet": articles[i].get("snippet", "")} for i in similar_indices]
    return recommendations

def translate_text(text, target_lang):
    translated = translator.translate(text, dest=target_lang)
    return translated.text

def login_page():
    st.title("🔒 Login")
    st.subheader("News Recommendation System in Regional Languages")  # Add this line
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    preferred_lang = st.selectbox("Select Preferred Language", LANGUAGES, format_func=lambda x: x.upper())
    if st.button("Login"):
        if authenticate_user(username, password):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.session_state["preferred_lang"] = preferred_lang
            st.experimental_rerun()
        else:
            st.error("Invalid username or password.")

def signup_page():
    st.title("📝 Sign Up")
    st.subheader("News Recommendation System in Regional Languages")  # Add this line
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    preferred_lang = st.selectbox("Select Preferred Language", LANGUAGES, format_func=lambda x: x.upper())
    if st.button("Sign Up"):
        if register_user(username, password, preferred_lang):
            st.success("Sign up successful! Please log in.")
        else:
            st.error("Username already exists. Please choose a different one.")

def news_recommender():
    st.title(f"📰 Welcome, {st.session_state['username']}!")
    genre = st.selectbox("Select a Genre:", GENRES[st.session_state["preferred_lang"]])
    if st.button("Fetch News"):
        try:
            with st.spinner("Fetching news articles..."):
                articles = fetch_news(API_KEY, query=genre)
                if not articles:
                    st.warning("No articles found for this genre. Try another one!")
                    return
                
                tfidf_matrix, _ = extract_features(articles)
                st.success(f"Found {len(articles)} articles for '{genre}'!")

                st.subheader("Top 10 News Articles:")
                for idx, article in enumerate(articles[:10]):
                    title = article.get("title", 'Untitled')
                    snippet = article.get("snippet", "No description available.")
                    link = article.get("link", "#")
                    st.write(f"**{idx + 1}.** {title} ([Read More]({link}))")
                    st.write(f"*Snippet:* {snippet}")

                # Further functionality can be added based on needs
        except Exception as e:
            st.error(f"Error occurred: {e}")

if __name__ == "__main__":
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"]:
        news_recommender()
    else:
        page = st.sidebar.radio("Select Page", ["Login", "Sign Up"])
        if page == "Login":
            login_page()
        else:
            signup_page()
