from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# k_nearest_neighbours_classifier()
# parameters:
#   neighbours : integer - The number of neighbours (closest training instances)
#       to be compared against
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
#   This function implements the K-Nearest Neighbours classification algorithm, and
#       trains it on the training data (training_instances_bow). Once the
#       algorithm has been trained, it then makes predictions using the test
#       data (test_instances_bow). The precision, recall and f_score are
#       generated using the test predictions and the actual labels. These values
#       are then returned.
def k_nearest_neighbours_classifier (neighbours, training_instances_bow, training_emotion_scores, test_instances_bow, test_emotion_scores):
    if neighbours:
        classifier = KNeighborsClassifier(n_neighbors=int(neighbours))
    else:
        classifier = KNeighborsClassifier()
    classifier.fit(training_instances_bow, training_emotion_scores)
    predicted_test_emotion_scores = classifier.predict(test_instances_bow)
    metrics = classification_report(test_emotion_scores, predicted_test_emotion_scores, digits=4, output_dict=True)
    return predicted_test_emotion_scores, metrics

# run()
# parameters:
#   modifier : integer - The number of neighbours (closest training instances)
#       to be compared against
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
#       the classification function (k_nearest_neighbours_classifier()) and
#       extracts the metrics from the returned data which are then returned to
#       the processor.data_split_bow_run() function.
def run (modifier, training_instances_bow, training_emotion_scores, test_instances_bow, test_emotion_scores):
    predictions, metrics = k_nearest_neighbours_classifier(modifier, training_instances_bow, training_emotion_scores, test_instances_bow, test_emotion_scores)
    return metrics
