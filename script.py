'''This file contains the code for nlp final project.
'''

# Import modules
import sys
from csv import reader
from typing import List
from nltk.tokenize import RegexpTokenizer
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation

# Determines which dataset use and how much to use :
# either 'HateSpeech' or 'KaggleTwitter' or 'TDavidson'
dataset_to_use = "TDavidson"
dataset_percentage = 1  # percentage range 1 to 100

# Initializes file path, column of csv file to parse and
# the delimiter for parsing
training_file = ""
column_to_parse = None
delimiter = ","
if dataset_to_use == "HateSpeech":
    training_file = "datasets/hate-speech/data.txt"
    delimiter = "\t"
    column_to_parse = 0
elif dataset_to_use == "KaggleTwitter":
    training_file = "datasets/kaggle-twitter/train.csv"
    column_to_parse = 2
elif dataset_to_use == "TDavidson":
    training_file = "datasets/t-davidson-hate-speech/labeled_data.csv"
    column_to_parse = 6
else:
    print("Invalid Dataset specified")
    sys.exit(1)


def preprocessing(running_lines: List[str]) -> List[List[str]]:
    """This function takes in the running test and return back the
    preprocessed text. Four tasks are done as part of this:
      1. lower word case
      2. remove stopwords
      3. remove punctuation
      4. Add - <s> and </s> for every sentence

    Args:
        running_lines (List[str]): list of lines

    Returns:
        List[List[str]]: list of sentences where each sentence is broken
                          into list of words.
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
        preprocessed_lines.append(processed_data)
    return preprocessed_lines


def parse_data(training_file_path: str, percentage: int,
               select_column: int, delimit: str) -> List[str]:
    """This function is used to parse input lines
    and returns a the provided percent of data.

    Args:
        lines (List[str]): list of lines
        percentage (int): percent of the dataset needed
        select_column (int): column to be selected from the dataset
        delimit (str): delimiter
    Returns:
        List[str]: lines (percentage of dataset)
    """
    percentage_sentences = []
    with open(training_file_path, "r", encoding="utf8",
              errors="ignore") as csvfile:
        read_sentences = []
        csv_reader = reader(csvfile, delimiter=delimit)
        # skipping header
        header = next(csv_reader)
        # line_length = len(list(csv_reader_copy))
        if header is not None:
            for row in csv_reader:
                read_sentences.append(row[select_column])
        end_of_data = int(len(read_sentences) * percentage * .01)
        percentage_sentences = read_sentences[0:end_of_data]
    return percentage_sentences


if __name__ == "__main__":
    # parse and preprocess the data
    sentences = preprocessing(parse_data(
        training_file, dataset_percentage, column_to_parse, delimiter))
    # verify the processed sentences
    for sentence in sentences:
        print(sentence)
    # This is the baseline classifier
    # This is the neural lm classifier
