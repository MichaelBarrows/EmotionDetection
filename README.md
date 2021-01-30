 <h1>Emotion Detection</h1>
<!-- <hr style="margin:10px 0;padding:0;"/> -->

This code is part of the research project titled "Iranian State-Sponsored Propaganda on Twitter: Exploring Methods for Automatic Classification and Analysis".

<h3>Project</h3>
<hr style="margin:10px 0;padding:0;"/>

This project aims to label Iranian state-sponsored propaganda for emotion and emotion using automated methods. Additionally, this project will evaluate various machine learning algorithms performance for correctly classifying emotion and emotion contained in the text based on the automated labelling.


<h3>About</h3>
<hr style="margin:10px 0;padding:0;"/>

This code runs the machine learning algorithms with the relevant data. This code contains functionality to:
  - Execute machine learning algorithms
    - with different features such as unigrams; bigrams; trigrams; unigrams & bigrams; and unigrams, bigrams & trigrams
  - generate baseline results, for each lower 10% (match percentage threshold) removed
  - generate initial experiment results
  - generate experimental roesults automatically based on the baseline and initial experiment results
  - partition the dataset into variou folds for evaluation
  - create features
  - store metrics
  - create results files
  - store all metrics for each experiment.

<h3>Prerequisites</h3>
<hr style="margin:10px 0;padding:0;"/>

**Note:** it is recommended that you create a virtual environment (the code was created for Python 3.6):

    mkvirtualenv --python=/usr/bin/python3.6 emotion_analysis
    workon emotion_analysis


To install the required dependencies:

    pip install -r requirements.txt

Copy `dataset.py.example` to `dataset.py`

    cp dataset.py.example dataset.py


in `dataset.py`:
- Add the absolute path to the `dataset` variable (folder containing the dataset files)
  - e.g. `/home/USER/emotion_labelling/data/`
- Add an absolute path to the `output_data` variable for storing output data
  - e.g. `/home/USER/emotion_analysis/output_data/`
- Modify other variables in the file as required

<h3>Requirements</h3>
<hr style="margin:10px 0;padding:0;"/>

The requirements for this code to run are detailed in `requirements.txt`.

<h3>Execution</h3>
<hr style="margin:10px 0;padding:0;"/>

- Within the Python virtual environment, run:
`python run_all.py`
