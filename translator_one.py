import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, TimeDistributed

# Dataset
english_to_french = [
    ("I am cold", "Je suis froid"),
    ("You are tired", "Tu es fatigué"),
    ("He is hungry", "Il a faim"),
    # Add more data here...
]

# Tokenization
english_sentences = [pair[0] for pair in english_to_french]
french_sentences = [pair[1] for pair in english_to_french]

tokenizer_eng = Tokenizer()
tokenizer_eng.fit_on_texts(english_sentences)
eng_vocab_size = len(tokenizer_eng.word_index) + 1
encoded_eng_sentences = tokenizer_eng.texts_to_sequences(english_sentences)

tokenizer_fra = Tokenizer()
tokenizer_fra.fit_on_texts(french_sentences)
fra_vocab_size = len(tokenizer_fra.word_index) + 1
encoded_fra_sentences = tokenizer_fra.texts_to_sequences(french_sentences)

# Padding
max_eng_length = max(len(sentence.split()) for sentence in english_sentences)
max_fra_length = max(len(sentence.split()) for sentence in french_sentences)

padded_eng_sentences = pad_sequences(encoded_eng_sentences, maxlen=max_eng_length, padding='post')
padded_fra_sentences = pad_sequences(encoded_fra_sentences, maxlen=max_fra_length, padding='post')

# Model
model = Sequential()
model.add(Embedding(eng_vocab_size, 256, mask_zero=True))
model.add(GRU(256, return_sequences=True))
model.add(TimeDistributed(Dense(fra_vocab_size, activation='softmax')))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Training
model.fit(padded_eng_sentences, padded_fra_sentences, epochs=10, validation_split=0.2)

# Translate function
def translate_sentence(sentence):
    encoded_sentence = tokenizer_eng.texts_to_sequences([sentence])
    padded_sentence = pad_sequences(encoded_sentence, maxlen=max_eng_length, padding='post')
    prediction = model.predict(padded_sentence)[0]
    predicted_sentence = ' '.join([tokenizer_fra.index_word[np.argmax(token)] for token in prediction])
    return predicted_sentence

# Qualitative validation
english_sentence = "I am cold"
translated_sentence = translate_sentence(english_sentence)
print(f"English: {english_sentence}")
print(f"French Translation: {translated_sentence}")
