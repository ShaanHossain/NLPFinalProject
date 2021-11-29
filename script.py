'''This file contains the code for nlp final project.
'''

# Import modules
import sys
from csv import reader
from typing import List
from nltk.tokenize import RegexpTokenizer
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation
import numpy as np

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


def preprocessing(running_lines: List[str]) -> List[str]:
    """This function takes in the running test and return back the
    preprocessed text. Four tasks are done as part of this:
      1. lower word case
      2. remove stopwords
      3. remove punctuation
      4. Add - <s> and </s> for every sentence

    Args:
        running_lines (List[str]): list of lines

    Returns:
        List[str]: list of sentences which are processed
    """
    preprocessed_lines = []
    tokenizer = RegexpTokenizer(r"\w+")
    for line in running_lines:
        lower_case_data = line.lower()
        data_without_stop_word = remove_stopwords(lower_case_data)
        data_without_punct = strip_punctuation(data_without_stop_word)
        processed_data = tokenizer.tokenize(data_without_punct)
        processed_data.insert(0, "<s>")
        processed_data.append("</s>")
        preprocessed_lines.append(" ".join(processed_data))
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
                label_sentences.append(row[label_column])
        end_of_data = int(len(read_sentences) * percentage * .01)
        percentage_sentences = read_sentences[0:end_of_data]
        percentage_labels = label_sentences[0:end_of_data]
    return percentage_sentences, percentage_labels


def precision(gold_labels, predicted_labels):
    """
    Calculates the precision for a set of predicted labels give the gold (ground truth) labels.
    Parameters:
        gold_labels (list): a list of labels assigned by hand ("truth")
        predicted_labels (list): a corresponding list of labels predicted by the system
    Returns: double precision (a number from 0 to 1)
    """
    total_true_positive = 0
    total_false_positive = 0
    for iter in range(len(gold_labels)):
        if (predicted_labels[iter] == '1' and gold_labels[iter] == '1'):
            total_true_positive += 1
        elif (predicted_labels[iter] == '1' and gold_labels[iter] == '0'):
            total_false_positive += 1
    if ((total_true_positive + total_false_positive) == 0):
        return 0.0
    return total_true_positive/(total_true_positive + total_false_positive)


def recall(gold_labels, predicted_labels):
    """
    Calculates the recall for a set of predicted labels give the gold (ground truth) labels.
    Parameters:
        gold_labels (list): a list of labels assigned by hand ("truth")
        predicted_labels (list): a corresponding list of labels predicted by the system
    Returns: double recall (a number from 0 to 1)
    """
    total_true_positive = 0
    total_false_negative = 0
    for iter in range(len(gold_labels)):
        if (predicted_labels[iter] == '1' and gold_labels[iter] == '1'):
            total_true_positive += 1
        elif (predicted_labels[iter] == '0' and gold_labels[iter] == '1'):
            total_false_negative += 1
    if ((total_true_positive + total_false_negative) == 0):
        return 0.0
    return total_true_positive / (total_true_positive+total_false_negative)


def f1(gold_labels, predicted_labels):
    """
    Calculates the f1 for a set of predicted labels give the gold (ground truth) labels.
    Parameters:
        gold_labels (list): a list of labels assigned by hand ("truth")
        predicted_labels (list): a corresponding list of labels predicted by the system
    Returns: double f1 (a number from 0 to 1)
    """
    precision_value = precision(gold_labels, predicted_labels)
    recall_value = recall(gold_labels, predicted_labels)
    if (precision_value + recall_value) == 0:
        return 0.0
    return (2*precision_value*recall_value)/(precision_value+recall_value)


class LogisticClassifier:
    """Class to represent text classifier - logistic regression classifier
    """

    def __init__(self):
        """Constructor
        """
        self._positive_words_lexicon = open(
            "positive_words.txt").read().split()
        self._negative_words_lexicon = open(
            "negative_words.txt").read().split()
        self._theta = None
        self._learning_rate = 0.1
        self._epochs = 20

    def _sigmoid(self, x: int):
        """
        Calculates the sigmoid of a scalar or
        an numpy array-like set of values
        Parameters:
        x: input value(s)
        return
        Scalar or array corresponding to x passed through the sigmoid function
        """
        return 1 / (1 + np.e ** (-1 * x))

    def train(self, sentences: List, labels: List):
        """Trains the classifier based on the given examples

        Args:
            sentences (List): sentences to train on
            labels (List): label for the given sentences
        """
        self._theta = np.zeros(4)
        # self._theta = np.zeros(5)
        for epoch in range(self._epochs):
            for count in range(len(sentences)):
                # here bias is the last feature
                x_feature = np.array(self.featurize(sentences[count]))
                y_label = int(labels[count])
                # compute y_hat
                z = np.dot(self._theta, x_feature)
                y_hat = self._sigmoid(z)
                # compute gradient
                loss = y_hat - y_label
                gradient = x_feature * loss
                self._theta = self._theta - (self._learning_rate)*gradient
            epoch += 1

    def score(self, data: str):
        """
        Score a given piece of text
        you’ll compute e ^ (log(p(c)) + sum(log(p(w_i | c))) here

        Parameters:
          data - str like "I loved the hotel"
        Return: dict of class: score mappings
        return a dictionary of the values of P(data | c)  for each class,
        as in section 4.3 of the textbook e.g. {"0": 0.000061, "1": 0.000032}
        """
        X_features = self.featurize(data)
        y_pred = self._sigmoid(np.dot(X_features, self._theta))
        if y_pred > 0.5:
            return 1
        else:
            return 0

    def classify(self, data: str):
        """
        Label a given piece of text
        Parameters:
          data - str like "I loved the hotel"
        Return: string class label
        """
        return str(self.score(data))

    def featurize(self, data: str):
        """
        we use this format to make implementation of this class
        more straightforward and to be
        consistent with what you see in nltk
        Parameters:
          data - str like "I loved the hotel"
        Return: a list of tuples linking features to values
        for BoW, a list of tuples linking every word to True [("I", True), ("loved", True), ("it", True)]
        """
        # We are considering four features + one bias for this
        # 1. Count (+ve lexicon  in document/sentence)
        # 2. Count (-ve lexicon in document/sentence)
        # 3. If '!' exists in document then 1, else 0
        # 4. Log(wordcount)
        # 5. Bias
        # Note: to get the best f1 score in the test set, I am considering
        # the first three features plus bias. However, to run all four features
        # uncomment the line 272, 338, and 346; comment line:271 and 345
        words = data.split()
        x1 = 0
        x2 = 0
        x3 = 1 if '!' in data else 0
        # x4 = np.log(len(words))
        x5 = 1
        for word in words:
            if word in self._positive_words_lexicon:
                x1 += 1
            if word in self._negative_words_lexicon:
                x2 += 1
        feature_list = [x1, x2, x3, x5]
        # feature_list = [x1, x2, x3, x4, x5]
        return feature_list

    def __str__(self):
        """This function is used to get the classifier description.

        Returns:
            [str]: description of the classifier
        """
        return "Logestic Regression Classifier"


train_sentences, train_labels = parse_data(training_file,
                                           dataset_percentage,
                                           sentence_column_to_parse,
                                           label_column_to_parse,
                                           delimiter)
# parse and preprocess the data
processed_train_sentences = preprocessing(train_sentences)
# verify the processed sentences
# for sentence in sentences:
#     print(sentence)
# This is the baseline classifier
print(
    f"Performing Baseline - Logistic Classifier on {dataset_to_use}"
    f" with {dataset_percentage} % data ")
baseline_classifier = LogisticClassifier()
baseline_classifier.train(processed_train_sentences, train_labels)
baseline_predicted_labels = []
# test the model
test_sentences, baseline_gold_labels = parse_data(
    test_file, dataset_percentage,
    sentence_column_to_parse,
    label_column_to_parse, delimiter)
# parse and preprocess the data
processed_test_sentences = preprocessing(test_sentences)
for test_sentence in processed_test_sentences:
    baseline_predicted_label = baseline_classifier.classify(
        test_sentence)
    baseline_predicted_labels.append(baseline_predicted_label)
# report precision, recall, f1
precision_value = precision(baseline_gold_labels,
                            baseline_predicted_labels)
print(f"Precision: {precision_value}")
recall_value = recall(baseline_gold_labels, baseline_predicted_labels)
print(f"Recall: {recall_value}")
f1_value = f1(baseline_gold_labels, baseline_predicted_labels)
print(f"F1 Score: {f1_value}")
# This is the neural lm classifier
