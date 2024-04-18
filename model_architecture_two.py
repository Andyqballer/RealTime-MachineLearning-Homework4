import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GRU, Dense, Attention, Concatenate, TimeDistributed

# Data
english_to_french = [
    ("I am cold", "Je suis froid"),
    ("You are tired", "Tu es fatigué"),
    ("He is hungry", "Il a faim"),
    ("She is happy", "Elle est heureuse"),
    ("We are friends", "Nous sommes amis"),
    ("They are students", "Ils sont étudiants"),
    ("The cat is sleeping", "Le chat dort"),
    ("The sun is shining", "Le soleil brille"),
    ("We love music", "Nous aimons la musique"),
    ("She speaks French fluently", "Elle parle couramment français"),
    ("He enjoys reading books", "Il aime lire des livres"),
    ("They play soccer every weekend", "Ils jouent au football chaque week-end"),
    ("The movie starts at 7 PM", "Le film commence à 19 heures"),
    ("She wears a red dress", "Elle porte une robe rouge"),
    ("We cook dinner together", "Nous cuisinons le dîner ensemble"),
    ("He drives a blue car", "Il conduit une voiture bleue"),
    ("They visit museums often", "Ils visitent souvent les musées"),
]

english_sentences = [pair[0] for pair in english_to_french]
french_sentences = [pair[1] for pair in english_to_french]

# Tokenization
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

# Problem 1: GRU-based Encoder-Decoder

# Model
input_layer = Input(shape=(max_eng_length,))
embedding_layer = Embedding(eng_vocab_size, 256)(input_layer)
gru_layer, gru_state = GRU(256, return_sequences=True, return_state=True)(embedding_layer)
decoder_output = GRU(256, return_sequences=True)(gru_layer)
output_layer = TimeDistributed(Dense(fra_vocab_size, activation='softmax'))(decoder_output)

model_1 = Model(inputs=input_layer, outputs=output_layer)
model_1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training
model_1.fit(padded_eng_sentences, padded_fra_sentences, epochs=50, validation_split=0.2)

# Problem 2: Attention Mechanism

# Model
input_layer = Input(shape=(max_eng_length,))
embedding_layer = Embedding(eng_vocab_size, 256)(input_layer)
gru_layer, gru_state = GRU(256, return_sequences=True, return_state=True)(embedding_layer)

attention_output = Attention()([gru_layer, gru_layer])
concatenated_output = Concatenate(axis=-1)([gru_layer, attention_output])

decoder_output = GRU(256, return_sequences=True)(concatenated_output)
output_layer = TimeDistributed(Dense(fra_vocab_size, activation='softmax'))(decoder_output)

model_2 = Model(inputs=input_layer, outputs=output_layer)
model_2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training
model_2.fit(padded_eng_sentences, padded_fra_sentences, epochs=50, validation_split=0.2)

# Translate function
def translate_sentence(model, tokenizer_eng, tokenizer_fra, sentence, max_length):
    encoded_sentence = tokenizer_eng.texts_to_sequences([sentence])
    padded_sentence = pad_sequences(encoded_sentence, maxlen=max_length, padding='post')
    prediction = model.predict(padded_sentence)[0]
    
    predicted_indices = [np.argmax(token) for token in prediction]
    predicted_words = [tokenizer_fra.index_word[idx] for idx in predicted_indices if idx > 0]
    
    predicted_sentence = ' '.join(predicted_words)
    return predicted_sentence

# Qualitative validation
for english_sentence in english_sentences:
    translated_sentence_1 = translate_sentence(model_1, tokenizer_eng, tokenizer_fra, english_sentence, max_eng_length)
    translated_sentence_2 = translate_sentence(model_2, tokenizer_eng, tokenizer_fra, english_sentence, max_eng_length)
    
    print(f"English: {english_sentence}")
    print(f"French Translation (Problem 1): {translated_sentence_1}")
    print(f"French Translation (Problem 2): {translated_sentence_2}\n")

# Evaluation
loss_1, accuracy_1 = model_1.evaluate(padded_eng_sentences, padded_fra_sentences)
loss_2, accuracy_2 = model_2.evaluate(padded_eng_sentences, padded_fra_sentences)

print(f"Problem 1 - Training Loss: {loss_1}, Accuracy: {accuracy_1}")
print(f"Problem 2 - Training Loss: {loss_2}, Accuracy: {accuracy_2}")
