import re
import pandas as pd
from hazm import Normalizer, word_tokenize, Stemmer, stopwords_list
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Data
df = pd.read_csv('Data.csv', header=None, names=['Text', 'Label'])

# Execute Topics
topics = df['Label'].unique()
topics_cleaned = [topic.strip().replace('"', '') for topic in topics]


# Preprocess text
def preprocess_text(text):
    normalizer = Normalizer()
    normalized_text = normalizer.normalize(text)

    # Clean text with Regex
    normalized_text = re.sub(r'\u200c', ' ', normalized_text)
    normalized_text = re.sub(r'[^\w\s]', '', normalized_text)
    normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()

    # Tokenization
    tokens = word_tokenize(normalized_text)

    # Remove Stopwords
    stopwords = stopwords_list()
    filtered_tokens = [token for token in tokens if token not in stopwords]

    # Stemming
    stemmer = Stemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return " ".join(stemmed_tokens)


# Preprocess all data
df['Processed_Text'] = df['Text'].apply(preprocess_text)

# Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Processed_Text'])


# Function to process the test file
def process_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            processed_text = preprocess_text(text)
            test_vector = vectorizer.transform([processed_text])

            # Calculate similarity with each topic
            similarities = cosine_similarity(test_vector, X)
            topic_index = similarities.argmax()  # Find the index with maximum similarity
            predicted_topic = df['Label'].iloc[topic_index]

            print(f"The file is most similar to the topic: {predicted_topic}")
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print("Error!", e)


# Run with test file
file_path = "test.txt"
process_file(file_path)
