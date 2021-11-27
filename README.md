# How to get the libaries installed correctly

https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/

## For MacOS:

Make sure to use python3 or python depending on what they are mapped to. Should be running python version 3.9.7

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

