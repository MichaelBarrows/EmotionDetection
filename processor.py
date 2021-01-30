from sklearn.model_selection import KFold
import bag_of_ngrams
import knn
import decision_tree
import random_forest
import naive_bayes
import linear_svm
import metric_storage

# data_split_bow_run()
# parameters:
#   algorithm : string - the name of the algorithm to be executed
#   modifier : integer/float - the modifier value (hyperparameter) for the
#       algorithm
#   n_folds : integer - the number of folds that the data will be split into
#   df : DataFrame - a dataframe containing the data to be split and used
#   n_grams : string - string detailing the number of n-grams to be used
# returns:
#   metric_id : int - a number representing the file metrics were stored in
#   emotion : list - a list containing the precision, recall and f1-score for
#       the emotion class
#   no_emotion : list - a list containing the precision, recall and f1-score for
#       the no_emotion class
#   weighted_avg : list - a list containing the precision, recall and f1-score
#       averaged across the classes
#   macro_avg : list - a list containing the precision, recall and f1-score
#       macro averaged across the classes
#   accuracy : int - the accuracy of the algorithm
# description:
#   This function implements the process of splitting data into folds for
#       training and testing, extracting the text and emotion labels,
#       converting the text to a bag of n-grams representation
#       (by calling bag_of_ngrams(n_grams)). This function then calls the
#       relevant algorithm to be used, returning the metrics for storage.
def data_split_bow_run (algorithm, emotion_str, modifier, n_folds, df, n_grams):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=12)
    metrics_dict = {emotion_str: {"precision": [], "recall": [], "f1-score": [], "support": [], "avg": {"precision": 0, "recall": 0, "f1-score": 0, "support": 0}},
                    "no_" + emotion_str: {"precision": [], "recall": [], "f1-score": [], "support": [], "avg": {"precision": 0, "recall": 0, "f1-score": 0, "support": 0}},
                    "macro avg": {"precision": [], "recall": [], "f1-score": [], "support": [], "avg": {"precision": 0, "recall": 0, "f1-score": 0, "support": 0}},
                    "weighted avg": {"precision": [], "recall": [], "f1-score": [], "support": [], "avg": {"precision": 0, "recall": 0, "f1-score": 0, "support": 0}},
                    "accuracy": {"list": [], "avg": 0}}

    for training_index, test_index in kf.split(df.index.tolist()):
        training_ids, training_texts, training_emotion_scores  = [], [], []
        test_ids, test_texts, test_emotion_scores = [], [], []
        for index, row in df.iterrows():
            if index in training_index:
                training_ids.append(index)
                training_texts.append(str(row.preprocessed_tweet_text))
                training_emotion_scores.append(str(row[emotion_str + "_str"]))
            elif index in test_index:
                test_ids.append(index)
                test_texts.append(str(row.preprocessed_tweet_text))
                test_emotion_scores.append(str(row[emotion_str + "_str"]))

        if n_grams == "unigrams":
            training_instances_bow, test_instances_bow = bag_of_ngrams.unigrams(training_texts, test_texts)
        elif n_grams == "bigrams":
            training_instances_bow, test_instances_bow = bag_of_ngrams.bigrams(training_texts, test_texts)
        elif n_grams == "trigrams":
            training_instances_bow, test_instances_bow = bag_of_ngrams.trigrams(training_texts, test_texts)
        elif n_grams == "unigrams_bigrams":
            training_instances_bow, test_instances_bow = bag_of_ngrams.unigrams_and_bigrams(training_texts, test_texts)
        elif n_grams == "unigrams_bigrams_trigrams":
            training_instances_bow, test_instances_bow = bag_of_ngrams.unigrams_bigrams_and_trigrams(training_texts, test_texts)
        else:
            return

        # call algorithm
        precision = []
        recall = []
        f_score = []
        if algorithm == "knn":
            metrics = knn.run(modifier, training_instances_bow, training_emotion_scores, test_instances_bow, test_emotion_scores)
        elif algorithm == "decision_tree":
            metrics = decision_tree.run(training_instances_bow, training_emotion_scores, test_instances_bow, test_emotion_scores)
        elif algorithm == "random_forest":
            metrics = random_forest.run(modifier, training_instances_bow, training_emotion_scores, test_instances_bow, test_emotion_scores)
        elif algorithm == "naive_bayes":
            metrics = naive_bayes.run(modifier, training_instances_bow, training_emotion_scores, test_instances_bow, test_emotion_scores)
        elif algorithm == "linear_svm":
            metrics = linear_svm.run(modifier, training_instances_bow, training_emotion_scores, test_instances_bow, test_emotion_scores)
        else:
            return

        for key in metrics:
            if key in metrics_dict:
                if key == "accuracy":
                    metrics_dict[key]["list"].append(metrics[key])
                    continue
                metrics_dict[key]["precision"].append(metrics[key]["precision"])
                metrics_dict[key]["recall"].append(metrics[key]["recall"])
                metrics_dict[key]["f1-score"].append(metrics[key]["f1-score"])
                metrics_dict[key]["support"].append(metrics[key]["support"])

    for key in metrics_dict:
        if key == "accuracy":
            metrics_dict[key]["avg"] = average(metrics_dict[key]["list"])
            continue
        metrics_dict[key]["avg"]["precision"] = average(metrics_dict[key]["precision"])
        metrics_dict[key]["avg"]["recall"] = average(metrics_dict[key]["recall"])
        metrics_dict[key]["avg"]["f1-score"] = average(metrics_dict[key]["f1-score"])
        metrics_dict[key]["avg"]["support"] = average(metrics_dict[key]["support"])
    metric_id = metric_storage.store_metrics(metrics_dict, emotion_str, algorithm, modifier, n_grams)
    emotion = [metrics_dict[emotion_str]["avg"]["precision"], metrics_dict[emotion_str]["avg"]["recall"], metrics_dict[emotion_str]["avg"]["f1-score"]]
    no_emotion = [metrics_dict["no_" + emotion_str]["avg"]["precision"], metrics_dict["no_" + emotion_str]["avg"]["recall"], metrics_dict["no_" + emotion_str]["avg"]["f1-score"]]
    weighted_avg = [metrics_dict["weighted avg"]["avg"]["precision"], metrics_dict["weighted avg"]["avg"]["recall"], metrics_dict["weighted avg"]["avg"]["f1-score"]]
    macro_avg = [metrics_dict["macro avg"]["avg"]["precision"], metrics_dict["macro avg"]["avg"]["recall"], metrics_dict["macro avg"]["avg"]["f1-score"]]
    accuracy = metrics_dict["accuracy"]["avg"]
    return metric_id, emotion, no_emotion, weighted_avg, macro_avg, accuracy

# average():
# parameters:
#   list : list - list of values to be averaged
# returns:
#   integer - the rounded (to 4 decimal places) average of all values in the
#       list
# description:
#   This function totals the items in the list and divides them by the length
#       of the list to generate an average which is then rounded to 4 decimal
#       places and returned.
def average (list):
    total = 0
    for item in list:
        total += item
    return round(total / len(list), 4)
