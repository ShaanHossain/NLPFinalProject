
# Instructions on running the code:

The BiLSTM model is stored in the bilstm.ipynb file.
In the second section of this file, you'll note dataset_to_use and dataset_percentage.
dataset_to_use should be either KaggleTwitter, HateSpeech, or TDavidson
dataset_percentage should be a float greater than 0 and less than or equal to 100

Clearing all of the cells and running all the cells will be sufficient for testing this model.
It will run 3 trials with each trial running for 3 epochs. 

It's important to make sure the glove.twitter.27B.50d.txt file is in the utils folder which are the GloVe embeddings from the citations in the report.

# How to get the libaries installed correctly

https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/

## For MacOS:

Make sure to use python3 or python depending on what they are mapped to. Should be running python version 3.8.12

Install Virtualenv

python3 -m pip install --user virtualenv

Create a virtual environment

`python3 -m venv venv`

Activate the environment

`source venv/bin/activate`

Run this to install all requirements from the requirements.txt file

`python3 -m pip install -r requirements.txt`

### To deactivate virtualenv

`deactivate`

### To install new requirements

Install requirements using pip3 install ...

Then 

`python3 -m pip freeze > requirements.txt`

To save the new requirements

## Libraries we installed:

- numpy
- nltk
- tensorflow
- keras
- gensim
- pandas
- sklearn
- matplotlib
- inflect
