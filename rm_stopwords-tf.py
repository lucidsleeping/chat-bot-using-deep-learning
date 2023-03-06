import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import text_to_word_sequence

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

with open('./New-Dataset/small-117M-k40.train.txt', 'r', encoding='utf-8') as f:
    input_text = f.read()

# Convert input text to lowercase
input_text = input_text.lower()

# Remove stop words
filtered_text = ' '.join([word for word in input_text.split() if word not in stop_words])

# Tokenize text
tokens = text_to_word_sequence(filtered_text)

with open('output_file.txt', 'w', encoding='utf-8') as f:
    f.write(' '.join(tokens))
