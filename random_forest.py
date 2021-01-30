from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# random_forest_classifier()
# parameters:
#   trees : integer/float - The number of decision trees to be used in the
#       random forest
#   features : integer/float = The number of features to be taken into
#       consideration
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
#   This function implements the Random Forest classification algorithm, and
#       trains it on the training data (training_instances_bow). Once the
#       algorithm has been trained, it then makes predictions using the test
#       data (test_instances_bow). The precision, recall and f_score are
#       generated using the test predictions and the actual labels. These values
#       are then returned.
def random_forest_classifier (trees, features, training_instances_bow, training_emotion_scores, test_instances_bow, test_emotion_scores):
    if trees and features:
        classifier = RandomForestClassifier(n_estimators=int(trees),
                                            max_features=int(features),
                                            n_jobs=100)
    else:
        classifier = RandomForestClassifier(n_jobs=100)
    classifier.fit(training_instances_bow, training_emotion_scores)
    predicted_test_emotion_scores = classifier.predict(test_instances_bow)
    metrics = classification_report(test_emotion_scores, predicted_test_emotion_scores, digits=4, output_dict=True)
    return predicted_test_emotion_scores, metrics

# run()
# parameters:
#   modifier : list - list containing values for the trees and features
#       parameters
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
#   This function takes the parameters required for the algorithm to run,
#       splits them into two separate variables (if present) and calls the
#       classification function (random_forest_classifier()) and extracts the
#       metrics from the returned data which are then returned to the
#       processor.data_split_bow_run() function.
def run (modifier, training_instances_bow, training_emotion_scores, test_instances_bow, test_emotion_scores):
    if modifier:
        trees, features = modifier[0], modifier[1]
    else:
        trees, features = None, None
    predictions, metrics = random_forest_classifier(trees, features, training_instances_bow, training_emotion_scores, test_instances_bow, test_emotion_scores)
    return metrics
