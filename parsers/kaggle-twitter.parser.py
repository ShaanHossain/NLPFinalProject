from csv import reader
from typing import List
from nltk.tokenize import RegexpTokenizer
from nltk import ngrams
from gensim.parsing.preprocessing import stem_text, remove_stopwords, strip_punctuation

column_to_parse = 2
training_file = "datasets/kaggle-twitter/train.csv"

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
  tokenizer = RegexpTokenizer(r'\w+')
  for line in running_lines:
    lower_case_data = line.lower()
    data_without_stop_word = remove_stopwords(lower_case_data)
    data_without_punct = strip_punctuation(data_without_stop_word)
    processed_data = tokenizer.tokenize(data_without_punct)
    processed_data.insert(0,"<s>")
    processed_data.append("</s>")
    preprocessed_lines.append(processed_data)
  return preprocessed_lines

def parse_data(training_file_path: str, percentage: int, select_column:int) -> List[str]:
  """This function is used to parse input lines
  and returns a the provided percent of data.

  Args:
      lines (List[str]): list of lines
      percentage (int): percent of the dataset needed
      select_column (int): column to be selected from the dataset
  Returns:
      List[str]: lines (percentage of dataset)
  """
  sentences = []
  percentage_sentences = []
  with open(training_file_path, "r", encoding="utf8", errors="ignore") as csvfile:
    csv_reader = reader(csvfile)
    #skipping header
    header = next(csv_reader)

    # line_length = len(list(csv_reader_copy))
   
    if header != None:
      for row in csv_reader:
        sentences.append(row[select_column])

    end_of_data = int(len(sentences) * percentage * .01)
    percentage_sentences = sentences[0:end_of_data]

  return percentage_sentences

sentences = preprocessing(parse_data(training_file, .1, column_to_parse))
for i in sentences:
    print(i)
