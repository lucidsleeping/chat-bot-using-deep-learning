import nltk
from tensorflow.keras.preprocessing.text import text_to_word_sequence

nltk.download('punkt')

with open('./New-Dataset/small-117M-k40.train.txt', 'r', encoding='utf-8') as f:
    input_text = f.read()

# Tokenize text
tokens = text_to_word_sequence(input_text)

# Create vocabulary
vocabulary = sorted(list(set(tokens)))

# Write vocabulary to file
with open('vocabulary.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(vocabulary))
