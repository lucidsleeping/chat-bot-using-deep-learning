import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load stop words
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Read input file
with open('./New-Dataset/small-117M-k40.train.txt', 'r', encoding='utf-8') as f:
    input_text = f.read()

# Convert input text to lowercase
input_text = input_text.lower()

# Tokenize text
tokens = word_tokenize(input_text)

# Filter out stop words
filtered_tokens = [word for word in tokens if not word in stop_words]

# Write tokenized text to output file
with open('stop_words.txt', 'w', encoding='utf-8') as f:
    f.write(' '.join(filtered_tokens))
