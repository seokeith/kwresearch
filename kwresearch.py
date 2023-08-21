import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from sklearn.linear_model import LinearRegression

def load_csv():
    path = input("Enter the path to the CSV file: ")
    return pd.read_csv(path)

def analyze_common_themes(data):
    data['tokens'] = data['keywords'].apply(word_tokenize)

    # Flatten list of tokens and calculate frequency
    all_tokens = [token for sublist in data['tokens'].tolist() for token in sublist]
    fdist = FreqDist(all_tokens)

    # Consider top 10 common tokens as themes
    return [item[0] for item in fdist.most_common(10)]

def train_models(data, common_themes):
    for theme in common_themes:
        data[theme] = data['tokens'].apply(lambda x: 1 if theme in x else 0)

    model_search_volume = LinearRegression().fit(data[common_themes], data['search volume'])
    model_keyword_difficulty = LinearRegression().fit(data[common_themes], data['keyword difficulty'])

    return model_search_volume, model_keyword_difficulty

def predict(data, common_themes, model_search_volume, model_keyword_difficulty):
    data['tokens'] = data['keywords'].apply(word_tokenize)

    for theme in common_themes:
        data[theme] = data['tokens'].apply(lambda x: 1 if theme in x else 0)

    data['predicted search volume'] = model_search_volume.predict(data[common_themes])
    data['predicted keyword difficulty'] = model_keyword_difficulty.predict(data[common_themes])

    return data

def main():
    print("Welcome to the Keyword Analyzer App!")

    # Step 1: Load first CSV and analyze
    data = load_csv()
    common_themes = analyze_common_themes(data)
    print(f"Common themes found: {', '.join(common_themes)}")

    # Step 2: Train prediction models
    model_search_volume, model_keyword_difficulty = train_models(data, common_themes)

    # Step 3: Load second CSV and predict
    second_batch = load_csv()
    predictions = predict(second_batch, common_themes, model_search_volume, model_keyword_difficulty)
    
    # Optionally, save the predictions to a CSV
    save = input("Do you want to save the predictions to a CSV? (yes/no): ").strip().lower()
    if save == 'yes':
        output_path = input("Enter the path to save the CSV: ")
        predictions.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

    print("Thank you for using the Keyword Analyzer App!")

if __name__ == "__main__":
    main()
