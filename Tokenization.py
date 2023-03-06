import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

# Read the text file
with open('output_file.txt', 'r', encoding='utf-8') as file: # ./New-Dataset/small-117M-k40.train.txt
    text = file.read()

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
sequences = tokenizer.texts_to_sequences([text])
tokenized_text = []
for sequence in sequences:
    tokenized_text += sequence

# Write the tokenized text to a file
with open('tokenized.txt', 'w', encoding='utf-8') as file:
    for token in tokenized_text:
        file.write(str(token) + ' ')
