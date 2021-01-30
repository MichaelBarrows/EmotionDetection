import processor

n_folds = 10

# run_knn_classification()
# parameters:
#   df : DataFrame - the dataframe containing the dataset
#   hyperparameter : int - the hyperparameter used to modify the algorithm
#   experiment_type : string - the string detailing the experiment type
#   n_grams : string - string detailing the number of n-grams to be used
# returns:
#   - : list - a list of the results ready to be appended to a list
# description:
#   This function calls the processor.data_split_bow_run() function in order to
#       execute the K-Nearest Neighbours classification algorithm. The results
#       of the execution are then returned returned.
def run_knn_classification (df, emotion_str, hyperparameter, experiment_type, n_grams):
    global n_folds
    metric_id, emotion, no_emotion, weighted_avg, macro_avg, accuracy = processor.data_split_bow_run("knn", emotion_str, hyperparameter, n_folds, df, n_grams)
    if not hyperparameter:
        hyperparameter = "default"
    return ["KNN",
            hyperparameter,
            weighted_avg[0],
            weighted_avg[1],
            weighted_avg[2],
            accuracy,
            experiment_type,
            metric_id,
            macro_avg[0],
            macro_avg[1],
            macro_avg[2],
            emotion[0],
            emotion[1],
            emotion[2],
            no_emotion[0],
            no_emotion[1],
            no_emotion[2]]

# run_decision_tree_classification()
# parameters:
#   df : DataFrame - the dataframe containing the dataset
#   experiment_type : string - the string detailing the experiment type
#   n_grams : string - string detailing the number of n-grams to be used
# returns:
#   - : list - a list of the results ready to be appended to a list
# description:
#   This function calls the processor.data_split_bow_run() function in order to
#       execute the Decision Tree classification algorithm. The results of the
#       execution are then returned returned.
def run_decision_tree_classification (df, emotion_str, experiment_type, n_grams):
    global n_folds
    metric_id, emotion, no_emotion, weighted_avg, macro_avg, accuracy = processor.data_split_bow_run("decision_tree", emotion_str, None, n_folds, df, n_grams)
    return ["Decision Tree",
            "default",
            weighted_avg[0],
            weighted_avg[1],
            weighted_avg[2],
            accuracy,
            experiment_type,
            metric_id,
            macro_avg[0],
            macro_avg[1],
            macro_avg[2],
            emotion[0],
            emotion[1],
            emotion[2],
            no_emotion[0],
            no_emotion[1],
            no_emotion[2]]

# run_random_forest_classification()
# parameters:
#   df : DataFrame - the dataframe containing the dataset
#   hyperparameters : list - the hyperparameters used to modify the algorithm
#   experiment_type : string - the string detailing the experiment type
#   n_grams : string - string detailing the number of n-grams to be used
# returns:
#   - : list - a list of the results ready to be appended to a list
# description:
#   This function calls the processor.data_split_bow_run() function in order to
#       execute the Random Forest classification algorithm. The results of the
#       execution are then returned returned.
def run_random_forest_classification (df, emotion_str, hyperparameters, experiment_type, n_grams):
    global n_folds
    metric_id, emotion, no_emotion, weighted_avg, macro_avg, accuracy = processor.data_split_bow_run("random_forest", emotion_str, hyperparameters, n_folds, df, n_grams)
    if not hyperparameters:
        hyperparameters = "default"
    return ["Random Forest",
            hyperparameters,
            weighted_avg[0],
            weighted_avg[1],
            weighted_avg[2],
            accuracy,
            experiment_type,
            metric_id,
            macro_avg[0],
            macro_avg[1],
            macro_avg[2],
            emotion[0],
            emotion[1],
            emotion[2],
            no_emotion[0],
            no_emotion[1],
            no_emotion[2]]

# run_naive_bayes_classification()
# parameters:
#   df : DataFrame - the dataframe containing the dataset
#   hyperparameter : int - the hyperparameter used to modify the algorithm
#   experiment_type : string - the string detailing the experiment type
#   n_grams : string - string detailing the number of n-grams to be used
# returns:
#   - : list - a list of the results ready to be appended to a list
# description:
#   This function calls the processor.data_split_bow_run() function in order to
#       execute the Naive Bayes classification algorithm. The results of the
#       execution are then returned returned.
def run_naive_bayes_classification (df, emotion_str, hyperparameter, experiment_type, n_grams):
    global n_folds
    metric_id, emotion, no_emotion, weighted_avg, macro_avg, accuracy = processor.data_split_bow_run("naive_bayes", emotion_str, hyperparameter, n_folds, df, n_grams)
    if not hyperparameter:
        hyperparameter = "default"
    return ["Naive Bayes",
            hyperparameter,
            weighted_avg[0],
            weighted_avg[1],
            weighted_avg[2],
            accuracy,
            experiment_type,
            metric_id,
            macro_avg[0],
            macro_avg[1],
            macro_avg[2],
            emotion[0],
            emotion[1],
            emotion[2],
            no_emotion[0],
            no_emotion[1],
            no_emotion[2]]

# run_linear_svm_classification()
# parameters:
#   df : DataFrame - the dataframe containing the dataset
#   hyperparameter : int - the hyperparameter used to modify the algorithm
#   experiment_type : string - the string detailing the experiment type
#   n_grams : string - string detailing the number of n-grams to be used
# returns:
#   - : list - a list of the results ready to be appended to a list
# description:
#   This function calls the processor.data_split_bow_run() function in order to
#       execute the Linear Support Vector Machine (SVM) classification
#       algorithm. The results of the execution are then returned returned.
def run_linear_svm_classification (df, emotion_str, hyperparameter, experiment_type, n_grams):
    global n_folds
    metric_id, emotion, no_emotion, weighted_avg, macro_avg, accuracy = processor.data_split_bow_run("linear_svm", emotion_str, hyperparameter, n_folds, df, n_grams)
    if not hyperparameter:
        hyperparameter = "default"
    return ["Linear SVM",
            hyperparameter,
            weighted_avg[0],
            weighted_avg[1],
            weighted_avg[2],
            accuracy,
            experiment_type,
            metric_id,
            macro_avg[0],
            macro_avg[1],
            macro_avg[2],
            emotion[0],
            emotion[1],
            emotion[2],
            no_emotion[0],
            no_emotion[1],
            no_emotion[2]]
