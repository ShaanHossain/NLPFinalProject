#import files
import sys
from csv import reader
from typing import List
from nltk.tokenize import RegexpTokenizer
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation
import numpy as np
import inflect
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import keras
from keras.models import Sequential
from keras.layers import Dropout, Dense, Embedding, LSTM, Bidirectional
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("Starting Neural Language System")

# Determines which dataset use and how much to use :
# HateSpeech: Column-0 : Sentence, Column-1 : Label [noHate-0, Hate-1]
# either 'HateSpeech' or 'KaggleTwitter' or 'TDavidson'
dataset_to_use = "HateSpeech"
dataset_percentage = 100  # percentage range 1 to 100

# Initializes file path, column of csv file to parse and
# the delimiter for parsing
training_file = ""
test_file = ""
sentence_column_to_parse = None
label_column_to_parse = None
lancaster = LancasterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
delimiter = ","
if dataset_to_use == "HateSpeech":
    training_file = "datasets/hate-speech/train.txt"
    test_file = "datasets/hate-speech/test.txt"
    delimiter = "\t"
    sentence_column_to_parse = 0
    label_column_to_parse = 1
elif dataset_to_use == "KaggleTwitter":
    training_file = "datasets/kaggle-twitter/train.csv"
    test_file = "datasets/kaggle-twitter/test.csv"
    sentence_column_to_parse = 2
    label_column_to_parse = 1
elif dataset_to_use == "TDavidson":
    training_file = "datasets/t-davidson-hate-speech/labeled_data.csv"
    # TODO: Update test path for this dataset
    # test_file = "datasets/kaggle-twitter/test.csv"
    sentence_column_to_parse = 6
    label_column_to_parse = 2
else:
    print("Invalid Dataset specified")
    sys.exit(1)

print(dataset_to_use + " Dataset Loaded")

def replace_numbers(sentence:List[str]) -> List[str]:
    """Replace all interger occurrences in list of tokenized words
    with textual representation"""
    p = inflect.engine()
    new_words = []  
    for word in sentence.split():
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return " ".join(new_words)

def stem_words(sentence: List[str]) -> List[str]:
    """Stems the given sentence

    Args:
        sentence (list): words to be stemmed

    Returns:
        str: stemmed sentence
    """
    stemmed_words = []
    for word in sentence.split():
        stemmed_words.append(lancaster.stem(word))
    return " ".join(stemmed_words)

def preprocessing(running_lines: List[str]) -> List[str]:
    """This function takes in the running test and return back the
    preprocessed text. Six tasks are done as part of this:
      1. lower word case
      2. remove stopwords
      3. remove punctuation
      4. convert numbers to texts
      5. perform stemming

    Args:
        sentence (List[str]): list of lines

    Returns:
        List[str]: list of sentences which are processed
    """
    preprocessed_lines = []
    tokenizer = RegexpTokenizer(r"\w+")
    for line in running_lines:
        # lower case
        lower_case_data = line.lower()
        # remove stop words
        data_without_stop_word = remove_stopwords(lower_case_data)
        # remove punctunation
        data_without_punct = strip_punctuation(data_without_stop_word)
        # replace numbers '1' to 'one'
        processed_data = replace_numbers(data_without_punct)
        # stem words
        processed_data = stem_words(processed_data)
        # add start and stop tags
        # processed_data.insert(0, "<s>")
        # processed_data.append("</s>")
        preprocessed_lines.append(processed_data)
    return preprocessed_lines

def parse_data(training_file_path: str, percentage: int,
               sentence_column: int, label_column: int,
               delimit: str):
    """This function is used to parse input lines
    and returns a the provided percent of data.

    Args:
        lines (List[str]): list of lines
        percentage (int): percent of the dataset needed
        sentence_column (int): sentence column from the dataset
        label_column (int): label column from the dataset
        delimit (str): delimiter
    Returns:
        List[str], List[str]: examples , labels -> [percentage of dataset]
    """
    percentage_sentences = []
    percentage_labels = []
    with open(training_file_path, "r", encoding="utf8",
              errors="ignore") as csvfile:
        read_sentences = []
        label_sentences = []
        csv_reader = reader(csvfile, delimiter=delimit)
        # skipping header
        header = next(csv_reader)
        # line_length = len(list(csv_reader_copy))
        if header is not None:
            for row in csv_reader:
                read_sentences.append(row[sentence_column])
                label_sentences.append(int(row[label_column]))
        end_of_data = int(len(read_sentences) * percentage * .01)
        percentage_sentences = read_sentences[0:end_of_data]
        percentage_labels = label_sentences[0:end_of_data]
    return percentage_sentences, percentage_labels

print("Parsing Data")

train_sentences, train_labels = parse_data(training_file,
                                           dataset_percentage,
                                           sentence_column_to_parse,
                                           label_column_to_parse,
                                           delimiter)
# parse and preprocess the data

print("Preprocessing Data")

processed_train_sentences = preprocessing(train_sentences)
# verify the processed sentences
# for sentence in sentences:
#     print(sentence)
# This is the baseline classifier
print(
    f"Performing Improved - BiLSTM on {dataset_to_use}"
    f" with {dataset_percentage} % data ")

def convert_sentence_word_embeddings(X_train_sentences:List[str]):
    """Converts the sentences into word embeddings.

    Args:
        X_train_sentences (List[str]): list of training sentences

    Returns:
        tuple: word embeddings for each sentence, vocab size and embedding dictionary
    """
    tokenizer = Tokenizer()
    text = np.array(X_train_sentences)
    tokenizer.fit_on_texts(X_train_sentences)
    # pickle.dump(tokenizer, open('text_tokenizer.pkl', 'wb'))
    # Uncomment above line to save the tokenizer as .pkl file 
    sequences = tokenizer.texts_to_sequences(text)
    word_index = tokenizer.word_index
    text = pad_sequences(sequences)
    print('Found %s unique tokens.' % len(word_index))
    indices = np.arange(text.shape[0])
    # np.random.shuffle(indices)
    text = text[indices]
    embeddings_dict = {}
    file_embeddings = open("utils/glove.twitter.27B.50d.txt", encoding="utf8")
    for embedding_line in file_embeddings:
        values = embedding_line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_dict[word] = coefs
    file_embeddings.close()
    print('Total %s word vectors.' % len(embeddings_dict))
    return (text, word_index, embeddings_dict)

print("Converting sentences to word embeddings")

X_train_Glove_s, word_index_s, embeddings_dict_s = convert_sentence_word_embeddings(processed_train_sentences)
embedding_size = len(X_train_Glove_s[0])
## Check function
# x_train_sample = ["Lorem Ipsum is simply dummy text of the printing and typesetting industry", "It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout"]
# X_train_Glove_s, word_index_s, embeddings_dict_s = convert_sentence_word_embeddings(x_train_sample)
# print("\n X_train_Glove_s \n ", X_train_Glove_s)
# print("\n Word index of the word testing is : ", word_index_s["industry"])
# print("\n Embedding for thw word want \n \n", embeddings_dict_s["want"])

'''
1. Number of BiLSTM layers
2. Number of nodes per BiLSTM layer
3. Number of nodes in dense layer right after BiLSTM/Dropout layers aka main computation matrix
4. Dropout for inputs to BiLSTM layers (percentage to drop inputs from previous layer)
5. Recurrent dropout in BiLSTM layers
6. Dropout in dropout layer right after BiLSTM layers  (percentage to drop from BiLSTM outputs)
7. Weight Initialization - Nope, set a random seed and fix it
8. Decay rate for optimizer (slowly lower learning rate)
9. Learning rate for optimizer
10. Which optimizer to use - Nope
11. Which loss function to use - Nope
12. Momentum - This only applies if we use SGD - Let's not
13. Epochs - Definitely
14. Batch size
15. Activation Function - Let's experiment ourselves then fix it
16. Embedding dimension

EMBEDDING_DIM=50, 
dropout=0.5, 
hidden_layer = 3, 
lstm_node = 32):

model = build_bilstm(
    word_index=word_index_s, 
    embeddings_dict=embeddings_dict_s, 
    embedding_dim=50, 
    num_hidden_layers=3, 
    num_nodes_per_hidden_layer=32,  
    num_nodes_final_fc_layer=256,
    input_dropout=0,
    recurrent_dropout=.2,
    output_dropout=.5,
    learning_rate=.1,
    max_sequence_length=embedding_size
    )

'''

def build_bilstm(
    word_index, 
    embeddings_dict,   
    embedding_dim,
    num_hidden_layers,
    num_nodes_per_hidden_layer,
    num_nodes_final_fc_layer,
    input_dropout,
    recurrent_dropout,
    output_dropout,
    learning_rate,
    max_sequence_length, 
    nclasses=2
    ):

    model = Sequential()
    # Make the embedding matrix using the embedding_dict
    embedding_matrix = np.random.random((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            if len(embedding_matrix[i]) != len(embedding_vector):
                print("could not broadcast input array from shape", str(len(embedding_matrix[i])),
                      "into shape", str(len(embedding_vector)), " Please make sure your"
                                                                " embedding_dim is equal to embedding_vector file ,GloVe,")
                exit(1)
            embedding_matrix[i] = embedding_vector
            
    # Add embedding layer
    model.add(Embedding(len(word_index) + 1,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_sequence_length,
                                trainable=True))
    # Add hidden layers 
    for i in range(0,num_hidden_layers):
        # Add a bidirectional lstm layer
        model.add(Bidirectional(LSTM(num_nodes_per_hidden_layer, return_sequences=True, recurrent_dropout=recurrent_dropout, dropout=input_dropout)))
        # Add a dropout layer after each lstm layer
        model.add(Dropout(output_dropout))
    model.add(Bidirectional(LSTM(num_nodes_per_hidden_layer, recurrent_dropout=recurrent_dropout, dropout=input_dropout)))
    model.add(Dropout(output_dropout))
    # Add the fully connected layer with 256 nurons and relu activation
    model.add(Dense(num_nodes_final_fc_layer, activation='relu'))
    # Add the output layer with softmax activation since we have 2 classes
    model.add(Dense(nclasses, activation='softmax'))
    # Compile the model using sparse_categorical_crossentropy
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=opt ,
                      metrics=['accuracy'])
    return model

print("Building Model!")
model = build_bilstm(
    word_index=word_index_s, 
    embeddings_dict=embeddings_dict_s, 
    embedding_dim=50, 
    num_hidden_layers=3, 
    num_nodes_per_hidden_layer=32,  
    num_nodes_final_fc_layer=256,
    input_dropout=0,
    recurrent_dropout=.2,
    output_dropout=.5,
    learning_rate=.1,
    max_sequence_length=embedding_size)

model.summary()

print("Training the model")

X_train, X_test, y_train, y_test = train_test_split(X_train_Glove_s, train_labels, test_size = 0.2)
data_history = model.fit(np.array(X_train), np.array(y_train),
                           validation_data=(np.array(X_test),np.array(y_test)),
                           epochs=20,
                           batch_size=128,
                           verbose=1)

def plot_graphs(axs, graph_index, history, string):
    axs[graph_index].plot(history.history[string])
    axs[graph_index].plot(history.history['val_'+string], '')
    # axs[graph_index].xlabel("Epochs")
    # axs[graph_index].ylabel(string)
    axs[graph_index].set(xlabel="Epochs", ylabel=string)
    axs[graph_index].legend([string, 'val_'+string])
    # axs[graph_index].show()

fig, axs = plt.subplots(2, figsize=(15, 15))
                        
plot_graphs(axs, 0, data_history, 'accuracy')
plot_graphs(axs, 1, data_history, 'loss')


'''

1. Number of BiLSTM layers
2. Number of nodes per BiLSTM layer
3. Number of nodes in dense layer right after BiLSTM/Dropout layers aka main computation matrix
4. Dropout for inputs to BiLSTM layers (percentage to drop inputs from previous layer)
5. Recurrent dropout in BiLSTM layers
6. Dropout in dropout layer right after BiLSTM layers  (percentage to drop from BiLSTM outputs)
7. Weight Initialization - Nope, set a random seed and fix it
8. Decay rate for optimizer (slowly lower learning rate)
9. Learning rate for optimizer
10. Which optimizer to use - Nope
11. Which loss function to use - Nope
12. Momentum - This only applies if we use SGD - Let's not
13. Epochs - Definitely
14. Batch size
15. Activation Function - Let's experiment ourselves then fix it
16. Embedding dimension

The ones we want to look at:
1. Number of BiLSTM layers
2. Number of nodes per BiLSTM layer
3. Epochs
4. Learning rate
5. Embedding dimension

Momentum explanation:

Momentum see here is a similar attempt to maintain a consistent direction. If we're taking smaller steps, it also makes sense to maintain a somewhat consistent heading through our space. We take a linear combination of the previous heading vector, and the newly-computed gradient vector, and adjust in that direction. For instance, if we have a momentum of 0.90, we will take 90% of the previous direction plus 10% of the new direction, and adjust weights accordingly -- multiplying that direction vector by the learning rate.

'''

# 1. NUMBER OF NODES AND HIDDEN LAYERS
# 2. NUMBER OF UNITS IN A DENSE LAYER
# 3. DROPOUT
# 4. WEIGHT INITIALIZATION
# 5. DECAY RATE
# 6. ACTIVATION FUNCTION
# 7. LEARNING RATE
# 8. MOMENTUM
# 9. NUMBER OF EPOCHS
# 10. BATCH SIZE
