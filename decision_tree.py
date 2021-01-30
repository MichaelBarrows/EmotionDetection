from sklearn import tree
from sklearn.metrics import classification_report

# decision_tree_classifier()
# parameters:
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
#   This function implements the Decision Tree classification algorithm, and
#       trains it on the training data (training_instances_bow). Once the
#       algorithm has been trained, it then makes predictions using the test
#       data (test_instances_bow). The precision, recall and f_score are
#       generated using the test predictions and the actual labels. These values
#       are then returned.
def decision_tree_classifier (training_instances_bow, training_emotion_scores, test_instances_bow, test_emotion_scores):
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(training_instances_bow, training_emotion_scores)
    predicted_test_emotion_scores = classifier.predict(test_instances_bow)
    metrics = classification_report(test_emotion_scores, predicted_test_emotion_scores, digits=4, output_dict=True)
    return predicted_test_emotion_scores, metrics

# run()
# parameters:
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
#       the classification function (decision_tree_classifier()) and extracts
#       the metrics from the returned data which are then returned to the
#       processor.data_split_bow_run() function.
def run (training_instances_bow, training_emotion_scores, test_instances_bow, test_emotion_scores):
    predictions, metrics = decision_tree_classifier(training_instances_bow, training_emotion_scores, test_instances_bow, test_emotion_scores)
    return metrics
