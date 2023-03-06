import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence

# Download required NLTK packages
nltk.download('punkt')
nltk.download('wordnet')

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Read input text
with open('output_file.txt', 'r', encoding='utf-8') as f:
    input_text = f.read()

# Convert input text to lowercase
input_text = input_text.lower()

# Remove stop words
stop_words = set(nltk.corpus.stopwords.words('english'))
filtered_text = ' '.join([word for word in input_text.split() if word not in stop_words])

# Tokenize text
tokens = text_to_word_sequence(filtered_text)

# Stem words
stemmed_tokens = [stemmer.stem(token) for token in tokens]

# Lemmatize words
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

# Write output to files
with open('stemmed.txt', 'w', encoding='utf-8') as f:
    f.write(' '.join(stemmed_tokens))

with open('lemmatized.txt', 'w', encoding='utf-8') as f:
    f.write(' '.join(lemmatized_tokens))
