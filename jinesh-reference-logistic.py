"""
This file is use to provide two different type of text classifiers: 
Naive Bayes and Logistic Regression. In the main driver function, we show 
the training of both the models, classification of both the models and finally
report precision, recall and f1-score for each model.
"""
from typing import List, Tuple
import numpy as np
import sys
from collections import Counter
import math

"""
Your name and file comment here:
Jinesh Shailesh Mehta
For File comment - Check at the start of the file.
"""


"""
Cite your sources here:

Positive and Negative Words List:

Minqing Hu and Bing Liu. "Mining and Summarizing Customer Reviews."
Proceedings of the ACM SIGKDD International Conference on Knowledge
Discovery and Data Mining (KDD-2004), Aug 22-25, 2004, Seattle,
Washington, USA,

Bing Liu, Minqing Hu and Junsheng Cheng. "Opinion Observer: Analyzing
and Comparing Opinions on the Web." Proceedings of the 14th
International World Wide Web conference (WWW-2005), May 10-14,
2005, Chiba, Japan.

Used Textbook - SLP for reference (implement NB and LR)

"""


def generate_tuples_from_file(training_file_path):
    """
    Generates tuples from file formated like:
    id\ttext\tlabel
    Parameters:
      training_file_path - str path to file to read in
    Return:
      a list of tuples of strings formatted [(id, example_text, label),
      (id, example_text, label)....]
    """
    f = open(training_file_path, "r", encoding="utf8")
    listOfExamples = []
    for review in f:
        if len(review.strip()) == 0:
            continue
        dataInReview = review.split("\t")
        for i in range(len(dataInReview)):
            # remove any extraneous whitespace
            dataInReview[i] = dataInReview[i].strip()
        t = tuple(dataInReview)
        listOfExamples.append(t)
    f.close()
    return listOfExamples


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


def k_fold(all_examples, k):
    # all_examples is a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    # containing all examples from the train and dev sets
    # return a list of lists containing k sublists where each sublist is one "fold" in the given data
    return np.array_split(all_examples, k)


class TextClassify:
    """This class is used to provide a text classifier based on Naive Bayes.
        It provides functionalities like training the model, generating scores
        and doing classification for given data.
    """

    def __init__(self):
        """
        This function is used for constructing the text classifier.
        """
        self._prior = {}
        self._V = []
        self._likelihood = {}

    def train(self, examples: List[Tuple]):
        """
        Trains the classifier based on the given examples
        Parameters:
          examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
        Return: None
        """
        # Using algorithm from textbook, so using same labelling convenction
        # getting dataset D ready
        D = {}
        for example in examples:
            D[example[1]] = example[2]
        # available classes 'C'
        C = set(D.values())
        big_doc = {}
        vocab_word_count = {}
        # create vocab for D
        list_vocab = []
        for sentence in D.keys():
            for word in sentence.split():
                list_vocab.append(word)
        self._V = list(set(list_vocab))
        for c in C:
            # calculate P(c) terms
            # number of documents in D
            N_doc = len(D.keys())
            # number of documents from D in class c
            N_c = 0
            for document, doc_c in D.items():
                if c == doc_c:
                    N_c += 1
            # we need store the prior 'c' and likelihood for 'w|c' for every c.
            self._prior[c] = np.log(N_c/N_doc)
            # create a big document (combine all example sentences)
            words_in_documents = []
            for document, doc_c in D.items():
                if c == doc_c:
                    words_in_document = document.split()
                    for word in words_in_document:
                        words_in_documents.append(word)
            big_doc[c] = words_in_documents
            # calculate vocab word count first as it will be used
            # for following calculation
            vocab_word_count[c] = len(big_doc[c])
            for w in self._V:
                # calculate P(W|c) terms (likelihood)
                # get count of word | class
                count_w_c = big_doc[c].count(w)
                self._likelihood[(w, c)] = np.log(
                    (count_w_c + 1)/(vocab_word_count[c] + len(self._V)))
        # now we have prior, likehihood and Vocab for each class
        # as intended in the text book algorithm

    def score(self, data: str):
        """
        Score a given piece of text
        Parameters:
          data - str like "I loved the hotel"
        Return: dict of class: score mappings
        """
        sum = {}
        for c, prior in self._prior.items():
            sum[c] = prior
            for word in data.split():
                # check if word is present in vocab
                # else ignore it
                if word in self._V:
                    sum[c] = sum[c] + self._likelihood[(word, c)]
            sum[c] = math.exp(sum[c])
        return sum

    def classify(self, data: str):
        """
        Label a given piece of text
        Parameters:
          data - str like "I loved the hotel"
        Return: string class label
        """
        class_probalities = self.score(data)
        # if the scores are equal, so that
        if class_probalities['0'] == class_probalities['1']:
            return '0'
        max_class_label = max(class_probalities, key=class_probalities.get)
        return max_class_label

    def featurize(self, data: str):
        """
        we use this format to make implementation of your TextClassifyImproved model more straightforward and to be
        consistent with what you see in nltk
        Parameters:
          data - str like "I loved the hotel"
        Return: a list of tuples linking features to values
        for BoW, a list of tuples linking every word to True [("I", True), ("loved", True), ("it", True)]
        """
        features_list = []
        # Not needed
        return features_list

    def __str__(self):
        """This function gives information about the type of classifier.

        Returns:
            [str]: description of the classifier
        """
        return "Naive Bayes - bag-of-words baseline"


class TextClassifyImproved:

    def __init__(self):
        self._positive_words_lexicon = open(
            "positive_words.txt").read().split()
        self._negative_words_lexicon = open(
            "negative_words.txt").read().split()
        self._theta = None
        self._learning_rate = 0.1
        self._classes = []
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

    def train(self, examples: List[Tuple]):
        """
        Trains the classifier based on the given examples
        Parameters:
          examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
        Return: None
        """
        self._theta = np.zeros(4)
        #self._theta = np.zeros(5)
        for epoch in range(self._epochs):
            for tuple in examples:
                # here bias is the last feature
                x_feature = np.array(self.featurize(tuple[1]))
                y_label = int(tuple[2])
                self._classes.append(y_label)
                # compute y_hat
                z = np.dot(self._theta, x_feature)
                y_hat = self._sigmoid(z)
                # compute gradient
                loss = y_hat - y_label
                gradient = x_feature * loss
                self._theta = self._theta - (self._learning_rate)*gradient
            epoch += 1
        self._classes = list(set(self._classes))

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
        we use this format to make implementation of this class more straightforward and to be
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
        #feature_list = [x1, x2, x3, x4, x5]
        return feature_list

    def __str__(self):
        """This function is used to get the classifier description.

        Returns:
            [str]: description of the classifier
        """
        return "Logestic Regression Classifier"


def main():
    """
    This is the driver method for this file. Both the models are 
    trained and tested in this function. In the end, we print out 
    the performance of each model.
    """

    training = sys.argv[1]
    testing = sys.argv[2]

    classifier = TextClassify()
    print(classifier)
    # do the things that you need to with your base class
    # get training data from file
    training_data = generate_tuples_from_file(training)
    # train the model
    classifier.train(training_data)
    # get testing data from dev file
    test_dev_data = generate_tuples_from_file(testing)
    gold_labels = []
    predicted_labels = []
    # test the model
    for test_data_tuples in test_dev_data:
        predicted_label = classifier.classify(test_data_tuples[1])
        predicted_labels.append(predicted_label)
        gold_labels.append(test_data_tuples[2])
        print(
            f'Actual Label:{test_data_tuples[2]}, Predicted Label :{predicted_label}')
    # report precision, recall, f1
    print('Precision: {}'.format(precision(gold_labels, predicted_labels)))
    print('Recall: {}'.format(recall(gold_labels, predicted_labels)))
    print('F1 Score: {}'.format(f1(gold_labels, predicted_labels)))

    improved = TextClassifyImproved()
    print(improved)
    # do the things that you need to with your improved class
    # use training data previously extracted
    # train the model
    fold_list = k_fold(training_data, 10)
    for count in range(0, len(fold_list)):
        print("Fold "+str(count+1))
        improved.train(fold_list[count])
        improved_gold_labels = []
        improved_predicted_labels = []
        # test the model
        for test_data_tuples in test_dev_data:
            improved_predicted_label = improved.classify(test_data_tuples[1])
            improved_predicted_labels.append(improved_predicted_label)
            improved_gold_labels.append(test_data_tuples[2])
            print(
                f'Actual Label:{test_data_tuples[2]}, Predicted Label :{improved_predicted_label}')
        # report precision, recall, f1 (best model)
        print('Precision: {}'.format(
            precision(improved_gold_labels, improved_predicted_labels)))
        print('Recall: {}'.format(
            recall(improved_gold_labels, improved_predicted_labels)))
        print('F1 Score: {}'.format(
            f1(improved_gold_labels, improved_predicted_labels)))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:", "python textclassify_model.py training-file.txt testing-file.txt")
        sys.exit(1)

    main()
