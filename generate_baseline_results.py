import helpers
import dataset as ds
import pandas as pd
import run

# get_baseline_results()
# parameters:
#   data : DataFrame - the data containing the data for processing
#   mpt : int - the match percentage threshold to be retained of the dataset
#   output_folder : string - the folder path for storing CSV files
#   n_grams : string - string detailing the number of n-grams to be used
# returns:
#   None
# description:
#   This function creates baseline results for each of the five machine learning
#       algorithms by removing part of the dataset with a
#       words_matched_percentage value lower than the mpt variable. The results
#       are then stored in a list, converted to a dataframe and stored in  a CSV
#       file.
def get_baseline_results (data, emotion, output_folder, n_grams):
    print("--- " + str(emotion) + " ---")
    results = []

    results.append(run.run_knn_classification(data, emotion, None, "baseline", n_grams))
    results.append(run.run_decision_tree_classification(data, emotion, "baseline", n_grams))
    results.append(run.run_linear_svm_classification(data, emotion, None, "baseline", n_grams))
    results.append(run.run_naive_bayes_classification(data, emotion, None, "baseline", n_grams))
    results.append(run.run_random_forest_classification(data, emotion, None, "baseline", n_grams))
    columns = ["algorithm",
            "hyperparameter",
            "weighted_avg_precision",
            "weighted_avg_recall",
            "weighted_avg_f1-score",
            "accuracy",
            "experiment_type",
            "metric_dump_id",
            "macro_avg_precision",
            "macro_avg_recall",
            "macro_avg_f1-score",
            emotion + "_precision",
            emotion + "_recall",
            emotion + "_f1-score",
            "no_" + emotion + "_precision",
            "no_" + emotion + "_recall",
            "no_" + emotion + "_f1-score"]
    results_df = pd.DataFrame(results, columns=columns)
    helpers.dataframe_to_csv(results_df, output_folder + emotion + "_results.csv")
    print(results_df[["algorithm", "hyperparameter", "macro_avg_f1-score", "accuracy", emotion + "_f1-score", "no_" + emotion + "_f1-score"]])


# negation_not_handled()
# parameters:
#   folder : string - the output folder
#   n_grams : string - string detailing the number of n-grams to be used
# returns:
#   None
# description:
#   This function loads the dataset (negation not handled), and calls the
#       get_baseline_results() function with varying values for the match
#       percentage threshold.
def execute (folder, n_grams):
    data = helpers.load_dataset(ds.dataset + ds.file)
    for emotion in ds.emotion_list:
        get_baseline_results(data, emotion, folder, n_grams)
