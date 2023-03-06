import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from transformer import TransformerBlock


def transformer_model(max_len, vocab_size):
    inputs = Input(shape=(max_len,))
    embedding_layer = tf.keras.layers.Embedding(vocab_size, 128)(inputs)
    x = embedding_layer
    transformer_block = TransformerBlock(128, 8, 32, 128)
    x = transformer_block(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = Dense(vocab_size, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(lr=1e-4)
    model.compile(loss=SparseCategoricalCrossentropy(), optimizer=optimizer)

    return model


# Load preprocessed dataset
with open('tokenized.txt', 'r', encoding='utf-8') as f:
    input_text = f.read()

# Tokenize input text
tokenizer = TextVectorization(output_sequence_length=40)
tokenizer.adapt([input_text])
input_text = tokenizer(input_text)
max_len = input_text.shape[1]

# Create and train model
model = transformer_model(max_len, len(tokenizer.word_index)+1)
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
history = model.fit(input_text, input_text, batch_size=32, epochs=50, callbacks=[early_stopping])

# Save model and tokenizer
model.save('fiction_gen_model.h5')
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
