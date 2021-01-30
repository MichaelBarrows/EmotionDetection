from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# linear_svm_classifier()
# parameters:
#   c_value : integer/float - hyperparameter
#   training_instances_bow : matrix - a representation of the frequency of words
#       in each training instance in reference to all words in all training
#       instances
#   training_emotion_scores : list - list of emotion labels for the training
#       data
#   test_instances_bow : matrix - a representation of the frequency of words in
#       each test instance in reference to all words in all test instances
#   test_emotion_scores : list - list of emotion labels for the test data
# returns:
#   predicted_test_emotion_scores : list - list of the predictions made by the
#       classifier
#   metrics : dict - a dictionary containing results metrics
# description:
#   This function implements the Linear SVM classification algorithm, and
#       trains it on the training data (training_instances_bow). Once the
#       algorithm has been trained, it then makes predictions using the test
#       data (test_instances_bow). The precision, recall and f_score are
#       generated using the test predictions and the actual labels. These values
#       are then returned.
def linear_svm_classifier (c_value, training_instances_bow, training_emotion_scores, test_instances_bow, test_emotion_scores):
    if c_value:
        classifier = LinearSVC(C=float(c_value), max_iter=10000)
    else:
        classifier = LinearSVC(max_iter=10000)
    classifier.fit(X=training_instances_bow, y=training_emotion_scores)
    predicted_test_emotion_scores = classifier.predict(test_instances_bow)
    metrics = classification_report(test_emotion_scores, predicted_test_emotion_scores, digits=4, output_dict=True)
    return predicted_test_emotion_scores, metrics

# run()
# parameters:
#   modifier : integer/float - hyperparameter
#   training_instances_bow : matrix - a representation of the frequency of words
#       in each training instance in reference to all words in all training
#       instances
#   training_emotion_scores : list - list of emotion labels for the training
#       data
#   test_instances_bow : matrix - a representation of the frequency of words in
#       each test instance in reference to all words in all test instances
#   test_emotion_scores : list - list of emotion labels for the test data
# returns:
#   metrics : dict - a dictionary containing results metrics
# description:
#   This function takes the parameters required for the algorithm to run, calls
#       the classification function (linear_svm_classifier()) and extracts the
#       metrics from the returned data which are then returned to the
#       processor.data_split_bow_run() function.
def run (modifier, training_instances_bow, training_emotion_scores, test_instances_bow, test_emotion_scores):
    predictions, metrics = linear_svm_classifier(modifier, training_instances_bow, training_emotion_scores, test_instances_bow, test_emotion_scores)
    return metrics
