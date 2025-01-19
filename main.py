import re
import pandas as pd
from hazm import Normalizer, word_tokenize, Stemmer, stopwords_list

# persica
df = pd.read_csv('Data.csv', header=None, names=['Text', 'Label'])

# Execute Topics
topics = df['Label'].unique()
topics_cleaned = [topic.strip().replace('"', '') for topic in topics]


def preprocess_text(text):
    # Normalization
    normalizer = Normalizer()
    normalized_text = normalizer.normalize(text)

    # RegEx
    normalized_text = re.sub(r'\u200c', ' ', normalized_text)  # Replace zero-width space with space
    normalized_text = re.sub(r'[^\w\s]', '', normalized_text)  # Remove punctuation
    normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()  # Remove extra spaces and leading/trailing spaces

    # Tokenization
    tokens = word_tokenize(normalized_text)

    # Delete Stopwords
    stopwords = stopwords_list()
    filtered_tokens = [token for token in tokens if token not in stopwords]

    # Delete Empty Token
    filtered_tokens = [token.strip() for token in filtered_tokens if token.strip()]

    # Stemming
    stemmer = Stemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    stemmed_tokens = [token for token in stemmed_tokens if token.strip()]

    return stemmed_tokens


def process_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            # print("Text:\n", text[:500])  # Show First 500 Char

            # Token
            processed_tokens = preprocess_text(text)
            # print("\nToken:\n", processed_tokens[:50])  # Show First 50 Token

            return processed_tokens
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print("Error!", e)


file_path = "test.txt"
processed_tokens = process_file(file_path)
print(topics_cleaned)
