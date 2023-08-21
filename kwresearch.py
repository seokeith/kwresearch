import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from sklearn.linear_model import LinearRegression

st.title('Keyword Analyzer App')

# Instructions for the user
st.markdown("""
### Instructions:
1. **First CSV Requirements**:
    - Columns: `keyword group`, `category 1`, `category 2`, `keywords`, `search volume`, `keyword difficulty`
    - This file should have known search volumes and keyword difficulties.
  
2. **Second CSV Requirements**:
    - Columns: `keywords`
    - This file is for predicting search volume and keyword difficulty based on the first CSV.

Upload the first CSV to begin the process.
""")
# Function to analyze common themes
def analyze_common_themes(data):
    data['tokens'] = data['keywords'].apply(word_tokenize)
    all_tokens = [token for sublist in data['tokens'].tolist() for token in sublist]
    fdist = FreqDist(all_tokens)
    return [item[0] for item in fdist.most_common(10)]

# Function to train prediction models
def train_models(data, common_themes):
    for theme in common_themes:
        data[theme] = data['tokens'].apply(lambda x: 1 if theme in x else 0)
    model_search_volume = LinearRegression().fit(data[common_themes], data['search volume'])
    model_keyword_difficulty = LinearRegression().fit(data[common_themes], data['keyword difficulty'])
    return model_search_volume, model_keyword_difficulty

# Function to make predictions
def predict(data, common_themes, model_search_volume, model_keyword_difficulty):
    data['tokens'] = data['keywords'].apply(word_tokenize)
    for theme in common_themes:
        data[theme] = data['tokens'].apply(lambda x: 1 if theme in x else 0)
    data['predicted search volume'] = model_search_volume.predict(data[common_themes])
    data['predicted keyword difficulty'] = model_keyword_difficulty.predict(data[common_themes])
    return data

uploaded_file = st.file_uploader("Choose your CSV file for analysis", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    common_themes = analyze_common_themes(data)
    st.write(f"Common themes found: {', '.join(common_themes)}")
    model_search_volume, model_keyword_difficulty = train_models(data, common_themes)

    second_file = st.file_uploader("Choose your second CSV file for prediction", type="csv")
    if second_file:
        second_batch = pd.read_csv(second_file)
        predictions = predict(second_batch, common_themes, model_search_volume, model_keyword_difficulty)
        st.write(predictions)
