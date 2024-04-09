import torch
import torch.nn as nn
import torch.optim as optim
from models.decoder_gru_model import DecoderGRU
from models.encoder_gru_model import EncoderGRU
import random

# Define the dataset
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
    ("The restaurant serves delicious food", "Le restaurant sert une délicieuse nourriture"),
    ("She studies mathematics at university", "Elle étudie les mathématiques à l'université"),
    ("We watch movies on Fridays", "Nous regardons des films le vendredi"),
    ("He listens to music while jogging", "Il écoute de la musique en faisant du jogging"),
    ("They travel around the world", "Ils voyagent autour du monde"),
    ("The book is on the table", "Le livre est sur la table"),
    ("She dances gracefully", "Elle danse avec grâce"),
    ("We celebrate birthdays with cake", "Nous fêtons les anniversaires avec un gâteau"),
    ("He works hard every day", "Il travaille dur tous les jours"),
    ("They speak different languages", "Ils parlent différentes langues"),
    ("The flowers bloom in spring", "Les fleurs éclosent au printemps"),
    ("She writes poetry in her free time", "Elle écrit de la poésie pendant son temps libre"),
    ("We learn something new every day", "Nous apprenons quelque chose de nouveau chaque jour"),
    ("The dog barks loudly", "Le chien aboie bruyamment"),
    ("He sings beautifully", "Il chante magnifiquement"),
    ("They swim in the pool", "Ils nagent dans la piscine"),
    ("The birds chirp in the morning", "Les oiseaux gazouillent le matin"),
    ("She teaches English at school", "Elle enseigne l'anglais à l'école"),
    ("We eat breakfast together", "Nous prenons le petit déjeuner ensemble"),
    ("He paints landscapes", "Il peint des paysages"),
    ("They laugh at the joke", "Ils rient de la blague"),
    ("The clock ticks loudly", "L'horloge fait tic-tac bruyamment"),
    ("She runs in the park", "Elle court dans le parc"),
    ("We travel by train", "Nous voyageons en train"),
    ("He writes a letter", "Il écrit une lettre"),
    ("They read books at the library", "Ils lisent des livres à la bibliothèque"),
    ("The baby cries", "Le bébé pleure"),
    ("She studies hard for exams", "Elle étudie dur pour les examens"),
    ("We plant flowers in the garden", "Nous plantons des fleurs dans le jardin"),
    ("He fixes the car", "Il répare la voiture"),
    ("They drink coffee in the morning", "Ils boivent du café le matin"),
    ("The sun sets in the evening", "Le soleil se couche le soir"),
    ("She dances at the party", "Elle danse à la fête"),
    ("We play music at the concert", "Nous jouons de la musique au concert"),
    ("He cooks dinner for his family", "Il cuisine le dîner pour sa famille"),
    ("They study French grammar", "Ils étudient la grammaire française"),
    ("The rain falls gently", "La pluie tombe doucement"),
    ("She sings a song", "Elle chante une chanson"),
    ("We watch a movie together", "Nous regardons un film ensemble"),
    ("He sleeps deeply", "Il dort profondément"),
    ("They travel to Paris", "Ils voyagent à Paris"),
    ("The children play in the park", "Les enfants jouent dans le parc"),
    ("She walks along the beach", "Elle se promène le long de la plage"),
    ("We talk on the phone", "Nous parlons au téléphone"),
    ("He waits for the bus", "Il attend le bus"),
    ("They visit the Eiffel Tower", "Ils visitent la tour Eiffel"),
    ("The stars twinkle at night", "Les étoiles scintillent la nuit"),
    ("She dreams of flying", "Elle rêve de voler"),
    ("We work in the office", "Nous travaillons au bureau"),
    ("He studies history", "Il étudie l'histoire"),
    ("They listen to the radio", "Ils écoutent la radio"),
    ("The wind blows gently", "Le vent souffle doucement"),
    ("She swims in the ocean", "Elle nage dans l'océan"),
    ("We dance at the wedding", "Nous dansons au mariage"),
    ("He climbs the mountain", "Il gravit la montagne"),
    ("They hike in the forest", "Ils font de la randonnée dans la forêt"),
    ("The cat meows loudly", "Le chat miaule bruyamment"),
    ("She paints a picture", "Elle peint un tableau"),
    ("We build a sandcastle", "Nous construisons un château de sable"),
    ("He sings in the choir", "Il chante dans le chœur"),
]

# Shuffle the dataset
random.shuffle(english_to_french)

# Split dataset into train and validation sets
split = int(0.8 * len(english_to_french))
train_data = english_to_french[:split]
val_data = english_to_french[split:]

# Define vocabulary
english_vocab = set()
french_vocab = set()
for pair in english_to_french:
    english_sentence, french_sentence = pair
    english_vocab.update(english_sentence.split())
    french_vocab.update(french_sentence.split())

# Add special tokens for padding, start, and end of sentence
PAD_token = 0
SOS_token = 1
EOS_token = 2
english_vocab.add('<PAD>')
english_vocab.add('<SOS>')
english_vocab.add('<EOS>')
french_vocab.add('<PAD>')
french_vocab.add('<SOS>')
french_vocab.add('<EOS>')

# Create word to index dictionaries
english_word_to_index = {word: i for i, word in enumerate(english_vocab)}
french_word_to_index = {word: i for i, word in enumerate(french_vocab)}

# Create index to word dictionaries
english_index_to_word = {i: word for word, i in english_word_to_index.items()}
french_index_to_word = {i: word for word, i in french_word_to_index.items()}

# Convert sentences to tensors of word indices
def sentence_to_tensor(sentence, vocab, word_to_index):
    indexes = [word_to_index[word] for word in sentence.split()]
    indexes.append(EOS_token)  # Append <EOS> token
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)



# Define the training function
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=20):
    encoder_hidden = torch.zeros(1, 1, encoder.hidden_size)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    _, encoder_hidden = encoder(input_tensor)

    decoder_input = torch.tensor([[SOS_token]])

    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()

        loss += criterion(decoder_output, target_tensor[di])
        if decoder_input.item() == EOS_token:
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

# Define the evaluation function
def evaluate(encoder, decoder, sentence, max_length=20):
    with torch.no_grad():
        input_tensor = sentence_to_tensor(sentence, english_vocab, english_word_to_index)
        input_length = input_tensor.size()[0]
        encoder_hidden = torch.zeros(1, 1, encoder.hidden_size)

        _, encoder_hidden = encoder(input_tensor)

        decoder_input = torch.tensor([[SOS_token]])
        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(french_index_to_word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return ' '.join(decoded_words)

# Initialize the models
encoder = EncoderGRU(len(english_vocab), 256)
decoder = DecoderGRU(256, len(french_vocab))

# Define the optimizers and criterion
encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)
criterion = nn.NLLLoss()

# Training the model
n_iters = 10000
print_every = 1000
plot_every = 100
all_losses = []
total_loss = 0

for iter in range(1, n_iters + 1):
    training_pair = random.choice(train_data)
    input_tensor = sentence_to_tensor(training_pair[0], english_vocab, english_word_to_index)
    target_tensor = sentence_to_tensor(training_pair[1], french_vocab, french_word_to_index)

    loss = train(input_tensor, target_tensor, encoder,
                 decoder, encoder_optimizer, decoder_optimizer, criterion)
    total_loss += loss

    if iter % print_every == 0:
        print('%d %d%% %.4f' % (iter, iter / n_iters * 100, total_loss / print_every))
        total_loss = 0

# Evaluation on validation set
def evaluate_randomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(val_data)
        print('English:', pair[0])
        print('Ground truth French:', pair[1])
        output_sentence = evaluate(encoder, decoder, pair[0])
        print('Generated French:', output_sentence)
        print('')

evaluate_randomly(encoder, decoder)
