from sklearn.feature_extraction.text import CountVectorizer

# unigrams()
# parameters:
#   training_texts : list - list containing the text for training the classifier
#   test_texts : list - list containing the text for testing the classifier
# returns:
#   training_instances_bow : array - the bag of words representation
#   test_instances_bow : array - the bag of words representation
# description:
#   This function converts a list of training texts and test texts into their
#       bag of words representations and returns them.
def unigrams (training_texts, test_texts):
    training_vectorizer = CountVectorizer()
    training_vectorizer.fit(training_texts)
    training_instances_bow = training_vectorizer.transform(training_texts)

    test_vectorizer = CountVectorizer(vocabulary=training_vectorizer.get_feature_names())
    test_vectorizer.fit(test_texts)
    test_instances_bow = test_vectorizer.fit_transform(test_texts)

    return training_instances_bow, test_instances_bow

# bigrams()
# parameters:
#   training_texts : list - list containing the text for training the classifier
#   test_texts : list - list containing the text for testing the classifier
# returns:
#   training_instances_bow : array - the bag of n-grams representation
#   test_instances_bow : array - the bag of n-grams representation
# description:
#   This function converts a list of training texts and test texts into their
#       bag of n-grams (bigrams) representations and returns them.
def bigrams (training_texts, test_texts):
    training_vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2))
    training_vectorizer.fit(training_texts)
    training_instances_bow = training_vectorizer.transform(training_texts)

    test_vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2), vocabulary=training_vectorizer.get_feature_names())
    test_vectorizer.fit(test_texts)
    test_instances_bow = test_vectorizer.fit_transform(test_texts)

    return training_instances_bow, test_instances_bow

# trigrams()
# parameters:
#   training_texts : list - list containing the text for training the classifier
#   test_texts : list - list containing the text for testing the classifier
# returns:
#   training_instances_bow : array - the bag of n-grams representation
#   test_instances_bow : array - the bag of n-grams representation
# description:
#   This function converts a list of training texts and test texts into their
#       bag of n-grams (trigrams) representations and returns them.
def trigrams (training_texts, test_texts):
    training_vectorizer = CountVectorizer(analyzer='word', ngram_range=(3, 3))
    training_vectorizer.fit(training_texts)
    training_instances_bow = training_vectorizer.transform(training_texts)

    test_vectorizer = CountVectorizer(analyzer='word', ngram_range=(3, 3), vocabulary=training_vectorizer.get_feature_names())
    test_vectorizer.fit(test_texts)
    test_instances_bow = test_vectorizer.fit_transform(test_texts)

    return training_instances_bow, test_instances_bow

# unigrams_and_bigrams()
# parameters:
#   training_texts : list - list containing the text for training the classifier
#   test_texts : list - list containing the text for testing the classifier
# returns:
#   training_instances_bow : array - the bag of n-grams representation
#   test_instances_bow : array - the bag of n-grams representation
# description:
#   This function converts a list of training texts and test texts into their
#       bag of n-grams (unigrams and bigrams) representations and returns them.
def unigrams_and_bigrams (training_texts, test_texts):
    training_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2))
    training_vectorizer.fit(training_texts)
    training_instances_bow = training_vectorizer.transform(training_texts)

    test_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2), vocabulary=training_vectorizer.get_feature_names())
    test_vectorizer.fit(test_texts)
    test_instances_bow = test_vectorizer.fit_transform(test_texts)

    return training_instances_bow, test_instances_bow

# unigrams_bigrams_and_trigrams()
# parameters:
#   training_texts : list - list containing the text for training the classifier
#   test_texts : list - list containing the text for testing the classifier
# returns:
#   training_instances_bow : array - the bag of n-grams representation
#   test_instances_bow : array - the bag of n-grams representation
# description:
#   This function converts a list of training texts and test texts into their
#       bag of n-grams (unigrams, bigrams and trigrams) representations and
#       returns them.
def unigrams_bigrams_and_trigrams (training_texts, test_texts):
    training_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 3))
    training_vectorizer.fit(training_texts)
    training_instances_bow = training_vectorizer.transform(training_texts)

    test_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 3), vocabulary=training_vectorizer.get_feature_names())
    test_vectorizer.fit(test_texts)
    test_instances_bow = test_vectorizer.fit_transform(test_texts)

    return training_instances_bow, test_instances_bow
