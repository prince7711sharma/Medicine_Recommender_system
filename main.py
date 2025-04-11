import streamlit as st
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse
import pickle

# Load Data
data = pd.read_csv('medicine.csv')

# Preprocessing
data['Description'] = data['Description'].apply(lambda x: x.split())
data['Reason'] = data['Reason'].apply(lambda x: x.split())
data['Description'] = data['Description'].apply(lambda x: [i.replace(" ", "") for i in x])
data['tags'] = data['Description'] + data['Reason']
data['tags'] = data['tags'].apply(lambda x: " ".join(x))
data['tags'] = data['tags'].apply(lambda x: x.lower())

# Stemming
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

data['tags'] = data['tags'].apply(stem)

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(data['tags']).toarray()

# Similarity matrix
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend_medicine(query):
    query = stem(query.lower())
    query_vec = cv.transform([query]).toarray()
    scores = cosine_similarity(query_vec, vectors)
    scores = list(enumerate(scores[0]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_5 = scores[:5]
    results = []
    for i in top_5:
        results.append(data.iloc[i[0]]['Drug_Name'])
    return results

# Streamlit UI
st.title("💊 Medicine Recommender App")
st.markdown("Type a condition, symptom, or keyword (like **acne**, **warts**, etc.) and get medicine suggestions.")

query = st.text_input("Enter symptoms / condition / keywords:")

#if query:
 #   st.subheader("Recommended Medicines:")
  #  recommendations = recommend_medicine(query)
  #  for i, med in enumerate(recommendations, 1):
   #     st.write(f"{i}. {med}")
##if query:
 #   st.subheader("Recommended Medicines:")
  #  recommendations = recommend_medicine(query)
   # for i, med in enumerate(recommendations, 1):
   #     # Encode medicine name for URL
   #     search_url = f"https://pharmeasy.in/search/all?name={urllib.parse.quote(med)}"
    #    st.markdown(f"**{i}. {med}**  \n[🛒 Buy Now]({search_url})", unsafe_allow_html=True)

if query:
    st.subheader("Recommended Medicines:")
    recommendations = recommend_medicine(query)
    for i, med in enumerate(recommendations, 1):
        search_url = f"https://pharmeasy.in/search/all?name={urllib.parse.quote(med)}"
        col1, col2 = st.columns([3, 1])  # Adjust width ratio as needed

        with col1:
            st.markdown(f"**{i}. {med}**")

        with col2:
            st.markdown(
                f"""
                <a href="{search_url}" target="_blank">
                    <button style="background-color:#4CAF50;
                                   border:none;
                                   color:white;
                                   padding:8px 16px;
                                   text-align:center;
                                   text-decoration:none;
                                   display:inline-block;
                                   font-size:14px;
                                   border-radius:6px;
                                   cursor:pointer;">
                        🛒 Buy Now 💊
                    </button>
                </a>
                """,
                unsafe_allow_html=True
            )

